import type { Page } from '@playwright/test'
import { expect, test } from '../../core/fixtures/base.fixture'

// A backfill's embedding calls skip the plugin pipeline, so they never reach
// the logs - the grey spend line next to the backfill counters is the only
// place an operator sees what indexing a window cost. The status endpoint is
// mocked, so this pins the rendering, not the job.

const configuredWarp = {
  configured: true,
  enabled: true,
  provider: 'openai',
  model: 'gpt-5.6-luna',
  max_iterations: 8,
  request_timeout_seconds: 120,
  history_retention_days: 30,
  embedding_provider: 'openai',
  embedding_model: 'text-embedding-3-small',
  embedding_dimension: 1536,
  log_vector_store_namespace: 'BifrostWarpLogs',
  semantic_search_threshold: 0.5,
  semantic_search_limit: 10,
  vector_store_connected: true,
  bifrost_mcp_available: true,
}

function completedJob(spend: { embedding_tokens?: number; embedding_cost?: number }) {
  return {
    id: 'warp-backfill-1',
    status: 'completed',
    start_time: '2026-09-17T00:00:00Z',
    end_time: '2026-09-24T00:00:00Z',
    total: 120,
    scanned: 120,
    indexed: 118,
    skipped: 2,
    failed: 0,
    ...spend,
    message: 'Scanned 120 log(s): 118 indexed, 2 skipped, 0 failed.',
  }
}

// The settings page opens a one-time notice about Warp's configuration tools
// for an enabled Warp; acknowledged up front so it does not cover the form.
async function acknowledgeMCPNotice(page: Page) {
  await page.addInitScript(() => localStorage.setItem('bifrost.warpBifrostMCPNoticeSeen', '1'))
}

// The warp flag is shared server state that other suites toggle in parallel,
// and it only gates the UI. Turning it on in this page's view of the flags
// leaves the server's copy alone, so no suite races another over it.
async function mockWarpBackfill(page: Page, job: object) {
  await acknowledgeMCPNotice(page)
  await page.route('**/api/feature-flags', async (route) => {
    if (route.request().method() !== 'GET') return route.continue()
    const response = await route.fetch()
    const body = (await response.json()) as { flags: { id: string; enabled: boolean }[] }
    body.flags = body.flags.map((flag) => (flag.id === 'warp' ? { ...flag, enabled: true } : flag))
    await route.fulfill({ response, json: body })
  })
  await page.route('**/api/warp/config', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(configuredWarp) }),
  )
  await page.route('**/api/warp/log-index/backfill/status**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(job) }),
  )
}

test.describe('Warp backfill spend', () => {
  test('shows embedding tokens and cost next to the backfill counters', async ({ page }) => {
    await mockWarpBackfill(page, completedJob({ embedding_tokens: 48210, embedding_cost: 0.00096 }))
    await page.goto('/workspace/config/warp')

    const status = page.getByTestId('warp-backfill-status')
    await expect(status).toBeVisible()
    await expect(status).toContainText('118 indexed · 2 skipped · 0 failed')
    // Sub-cent spend keeps four places rather than rounding down to "$0.00".
    await expect(page.getByTestId('warp-backfill-spend')).toHaveText('48,210 tokens · $0.0010')
  })

  test('shows tokens alone when the deployment cannot price them', async ({ page }) => {
    await mockWarpBackfill(page, completedJob({ embedding_tokens: 48210 }))
    await page.goto('/workspace/config/warp')

    await expect(page.getByTestId('warp-backfill-status')).toBeVisible()
    await expect(page.getByTestId('warp-backfill-spend')).toHaveText('48,210 tokens')
  })

  test('shows no spend line before any embedding call was made', async ({ page }) => {
    await mockWarpBackfill(page, completedJob({}))
    await page.goto('/workspace/config/warp')

    await expect(page.getByTestId('warp-backfill-status')).toBeVisible()
    await expect(page.getByTestId('warp-backfill-spend')).toHaveCount(0)
  })
})

// The status endpoint answers for the space the deployment is configured with,
// but the form can be edited off that space before anything is saved. The old
// run's "Completed" then sat under a model nothing has been indexed for, so
// the card follows the form: gone while the embedding space is edited, back
// when the fields return to the saved values.
test.describe('Warp backfill status and the embedding space', () => {
  test('hides a finished backfill while the embedding space is edited', async ({ page }) => {
    await mockWarpBackfill(page, completedJob({ embedding_tokens: 48210 }))
    await page.goto('/workspace/config/warp')

    const status = page.getByTestId('warp-backfill-status')
    await expect(status).toBeVisible()

    const dimension = page.getByTestId('warp-embedding-dimension-input')
    await dimension.fill('3072')
    await expect(status).toHaveCount(0)

    await dimension.fill(String(configuredWarp.embedding_dimension))
    await expect(status).toBeVisible()
  })
})

// The server refuses to change the embedding space while a backfill is in
// flight, but the moment the job finishes a save goes through - and the
// id-pinned poll the page uses during a run still answers for that job. So the
// old space's "Completed" could arrive after the new space was saved and sit
// on screen as a finished backfill of rows the new space never indexed.
test.describe('Warp backfill status after the embedding space is saved over', () => {
  test('drops a run whose terminal status lands after its space was replaced', async ({ page }) => {
    await acknowledgeMCPNotice(page)
    const runningJob = {
      id: 'warp-backfill-2',
      status: 'running',
      start_time: '2026-09-17T00:00:00Z',
      end_time: '2026-09-24T00:00:00Z',
      total: 120,
      scanned: 60,
      indexed: 60,
      skipped: 0,
      failed: 0,
    }
    const newSpace = { ...configuredWarp, embedding_dimension: 3072, log_vector_store_namespace: 'BifrostWarpLogsV2' }
    let savedConfig = configuredWarp
    await page.route('**/api/feature-flags', async (route) => {
      if (route.request().method() !== 'GET') return route.continue()
      const response = await route.fetch()
      const body = (await response.json()) as { flags: { id: string; enabled: boolean }[] }
      body.flags = body.flags.map((flag) => (flag.id === 'warp' ? { ...flag, enabled: true } : flag))
      await route.fulfill({ response, json: body })
    })
    await page.route('**/api/warp/config', async (route) => {
      if (route.request().method() === 'PUT') savedConfig = newSpace
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(savedConfig) })
    })
    await page.route('**/api/warp/log-index/backfill/status**', (route) => {
      const pinned = new URL(route.request().url()).searchParams.get('id') === runningJob.id
      // Once the new space is saved, the job is over: the pinned read still
      // reports it (by id, no space check), while the id-less read hides a run
      // of a space that is no longer configured.
      const job =
        savedConfig === configuredWarp
          ? runningJob
          : pinned
            ? { ...runningJob, status: 'completed', scanned: 120, indexed: 120 }
            : { status: 'idle' }
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(job) })
    })
    await page.goto('/workspace/config/warp')

    const status = page.getByTestId('warp-backfill-status')
    await expect(status).toContainText(/running/i)
    await expect(page.getByTestId('warp-backfill-cancel-btn')).toBeVisible()

    await page.getByTestId('warp-embedding-dimension-input').fill(String(newSpace.embedding_dimension))
    await page.getByTestId('warp-vector-namespace-input').fill(newSpace.log_vector_store_namespace)
    await page.getByTestId('warp-save-btn').click()

    // The pinned poll has delivered the terminal status once Cancel gives way
    // to Start. The old run must not have been kept as this space's result.
    await expect(page.getByTestId('warp-backfill-start-btn')).toBeVisible()
    await expect(status).toHaveCount(0)
  })
})

// The threshold is a 0-1 similarity on every vector store, and real topic
// matches score 0.575 to 0.735 on it, so a deployment that never set one
// starts at 0.5. A saved value is the operator's and is shown as saved.
test.describe('Warp semantic search threshold', () => {
  test('shows 0.5 when none is saved, and a saved value unchanged', async ({ page }) => {
    await acknowledgeMCPNotice(page)
    await page.route('**/api/feature-flags', async (route) => {
      if (route.request().method() !== 'GET') return route.continue()
      const response = await route.fetch()
      const body = (await response.json()) as { flags: { id: string; enabled: boolean }[] }
      body.flags = body.flags.map((flag) => (flag.id === 'warp' ? { ...flag, enabled: true } : flag))
      await route.fulfill({ response, json: body })
    })
    let threshold = 0
    await page.route('**/api/warp/config', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ...configuredWarp, semantic_search_threshold: threshold }),
      }),
    )

    await page.goto('/workspace/config/warp')
    await expect(page.getByTestId('warp-search-threshold-input')).toHaveValue('0.5')

    threshold = 0.62
    await page.reload()
    await expect(page.getByTestId('warp-search-threshold-input')).toHaveValue('0.62')
  })
})

// Warp keeps its log index in one of the vector stores configured in
// config.json (the primary vector_store or a vector_stores entry), chosen on
// the settings page. The picker appears only when there is a choice to make,
// and a store switched to starts empty, so the page says so before the save.

type ListedVectorStore = {
  name: string
  type: string
  primary: boolean
  status: 'connected' | 'failed' | 'disabled'
  error: string
  shared_with: string
  endpoint: string
  retention_days: number
  max_size_mb: number
}

function listedStore(name: string, type: string, primary: boolean, status: ListedVectorStore['status'] = 'connected', error = ''): ListedVectorStore {
  return { name, type, primary, status, error, shared_with: '', endpoint: '', retention_days: 0, max_size_mb: 0 }
}

const primaryStore = listedStore('default', 'weaviate', true)
const qdrantStore = listedStore('warp-qdrant', 'qdrant', false)
const failedStore = listedStore('warp-redis', 'redis', false, 'failed', 'dial tcp 127.0.0.1:6379: connection refused')

const warpOnPrimary = {
  ...configuredWarp,
  vector_store: '',
  connected_vector_store: { name: 'default', type: 'weaviate', primary: true },
}

// The warp flag is shared server state; turned on in this page's view only.
async function showWarpFlag(page: Page) {
  await page.route('**/api/feature-flags', async (route) => {
    if (route.request().method() !== 'GET') return route.continue()
    const response = await route.fetch()
    const body = (await response.json()) as { flags: { id: string; enabled: boolean }[] }
    body.flags = body.flags.map((flag) => (flag.id === 'warp' ? { ...flag, enabled: true } : flag))
    await route.fulfill({ response, json: body })
  })
}

/**
 * Mocks the Warp config, the store list and an idle backfill. A PUT answers
 * with `putResponse` when given; otherwise it is saved, and a vector_store in
 * it moves connected_vector_store to that store ("" is the primary).
 */
async function mockWarpVectorStores(
  page: Page,
  stores: ListedVectorStore[],
  options: { config?: object; putResponse?: { status: number; body: object } } = {},
) {
  await acknowledgeMCPNotice(page)
  await showWarpFlag(page)
  const puts: Record<string, unknown>[] = []
  let saved: object = options.config ?? warpOnPrimary
  await page.route('**/api/warp/config', async (route) => {
    if (route.request().method() === 'PUT') {
      const body = route.request().postDataJSON() as Record<string, unknown>
      puts.push(body)
      if (options.putResponse) {
        return route.fulfill({ status: options.putResponse.status, contentType: 'application/json', body: JSON.stringify(options.putResponse.body) })
      }
      if (typeof body.vector_store === 'string') {
        const chosen = body.vector_store
        const store = stores.find((s) => (chosen === '' ? s.primary : s.name === chosen))
        saved = {
          ...saved,
          vector_store: chosen,
          vector_store_connected: !!store,
          connected_vector_store: store ? { name: store.name, type: store.type, primary: store.primary } : null,
        }
      }
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(saved) })
  })
  await page.route('**/api/config/vector-stores', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ editable: true, vector_stores: stores }) }),
  )
  await page.route('**/api/warp/log-index/backfill/status**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'idle' }) }),
  )
  return puts
}

/** Opens the settings page and waits until the config and the store list have both rendered. */
async function openWarpSettings(page: Page) {
  await page.goto('/workspace/config/warp')
  // Set once the store list is rendered, an empty one included: a response that has arrived is
  // not yet a page that shows it, so checking the picker's absence before this proves nothing.
  await expect(page.getByTestId('warp-vector-store-status')).toHaveAttribute('data-vector-stores-loaded', 'true')
}

test.describe('Warp vector store picker', () => {
  for (const stores of [[], [primaryStore]]) {
    test(`is absent with ${stores.length} configured store(s)`, async ({ page }) => {
      await mockWarpVectorStores(page, stores)
      await openWarpSettings(page)

      await expect(page.getByTestId('warp-vector-store-select')).toHaveCount(0)
      await expect(page.getByTestId('warp-vector-store-status')).toContainText('default')
    })
  }

  test('lists every store, switches to another, and reports the new one after saving', async ({ page }) => {
    const puts = await mockWarpVectorStores(page, [primaryStore, qdrantStore, failedStore])
    await openWarpSettings(page)

    const select = page.getByTestId('warp-vector-store-select')
    await expect(select).toBeVisible()
    // "" is the primary store, shown by its name.
    await expect(select).toContainText('default')
    await expect(page.getByTestId('warp-vector-store-status')).toContainText('default')
    await expect(page.getByTestId('warp-vector-store-switch-note')).toHaveCount(0)

    await select.click()
    await expect(page.getByTestId('warp-vector-store-option-default')).toBeVisible()
    await expect(page.getByTestId('warp-vector-store-option-warp-qdrant')).toContainText('qdrant')
    // A store the server cannot reach is listed but cannot be picked.
    const failed = page.getByTestId('warp-vector-store-option-warp-redis')
    await expect(failed).toBeVisible()
    await expect(failed).toHaveAttribute('aria-disabled', 'true')

    await page.getByTestId('warp-vector-store-option-warp-qdrant').click()
    await expect(select).toContainText('warp-qdrant')
    const note = page.getByTestId('warp-vector-store-switch-note')
    await expect(note).toBeVisible()
    await expect(note).toContainText('warp-qdrant')

    const put = page.waitForResponse((r) => new URL(r.url()).pathname.endsWith('/api/warp/config') && r.request().method() === 'PUT')
    await page.getByTestId('warp-save-btn').click()
    const res = await put
    expect(res.status(), await res.text()).toBe(200)
    expect(puts).toHaveLength(1)
    expect(puts[0].vector_store).toBe('warp-qdrant')

    await expect(page.getByTestId('warp-vector-store-status')).toContainText('warp-qdrant')
    await expect(note).toHaveCount(0)
  })

  test('sends "" when switching back to the primary store', async ({ page }) => {
    const onQdrant = {
      ...configuredWarp,
      vector_store: 'warp-qdrant',
      connected_vector_store: { name: 'warp-qdrant', type: 'qdrant', primary: false },
    }
    const puts = await mockWarpVectorStores(page, [primaryStore, qdrantStore], { config: onQdrant })
    await openWarpSettings(page)

    await page.getByTestId('warp-vector-store-select').click()
    await page.getByTestId('warp-vector-store-option-default').click()
    await expect(page.getByTestId('warp-vector-store-switch-note')).toBeVisible()

    const put = page.waitForResponse((r) => new URL(r.url()).pathname.endsWith('/api/warp/config') && r.request().method() === 'PUT')
    await page.getByTestId('warp-save-btn').click()
    expect((await put).status()).toBe(200)
    expect(puts[0].vector_store).toBe('')
    await expect(page.getByTestId('warp-vector-store-status')).toContainText('default')
  })

  // An untouched picker leaves the stored choice alone: the field is omitted.
  test('omits vector_store from a save that did not change the store', async ({ page }) => {
    const puts = await mockWarpVectorStores(page, [primaryStore, qdrantStore])
    await openWarpSettings(page)

    await page.getByTestId('warp-search-limit-input').fill('12')
    const put = page.waitForResponse((r) => new URL(r.url()).pathname.endsWith('/api/warp/config') && r.request().method() === 'PUT')
    await page.getByTestId('warp-save-btn').click()
    expect((await put).status()).toBe(200)
    expect(puts).toHaveLength(1)
    expect('vector_store' in puts[0]).toBe(false)
  })

  test('shows the server refusal when a backfill is running', async ({ page }) => {
    await mockWarpVectorStores(page, [primaryStore, qdrantStore], {
      putResponse: { status: 409, body: { error: { message: 'a backfill is in progress' } } },
    })
    await openWarpSettings(page)

    await page.getByTestId('warp-vector-store-select').click()
    await page.getByTestId('warp-vector-store-option-warp-qdrant').click()
    await page.getByTestId('warp-save-btn').click()
    await expect(page.getByText('A backfill is in progress')).toBeVisible()
    // Nothing was saved, so the switch is still pending.
    await expect(page.getByTestId('warp-vector-store-switch-note')).toBeVisible()
  })

  test('shows why Warp has no store when the chosen one is unavailable', async ({ page }) => {
    const reason = 'Warp\'s vector store "warp-qdrant" could not be reached at startup'
    await mockWarpVectorStores(page, [primaryStore, listedStore('warp-qdrant', 'qdrant', false, 'failed', 'dial tcp: i/o timeout')], {
      config: {
        ...configuredWarp,
        vector_store: 'warp-qdrant',
        vector_store_connected: false,
        connected_vector_store: null,
        vector_store_error: reason,
      },
    })
    await openWarpSettings(page)

    await expect(page.getByTestId('warp-vector-store-status')).toHaveText(reason)
  })
})

// Against the real server: runs only where config.json configures at least two
// vector stores, and puts Warp back on the store it started on.
test.describe('Warp vector store switch on the server', () => {
  test('switches Warp to another connected store and reports it', async ({ page }) => {
    await acknowledgeMCPNotice(page)
    await showWarpFlag(page)

    const listedRes = await page.request.get('/api/config/vector-stores')
    expect(listedRes.status(), await listedRes.text()).toBe(200)
    const stores = ((await listedRes.json()) as { vector_stores: ListedVectorStore[] }).vector_stores
    test.skip(
      stores.length < 2,
      `the E2E server profile configures ${stores.length} vector store(s) (vector_store / vector_stores in config.json); switching Warp's store needs at least two`,
    )

    const configRes = await page.request.get('/api/warp/config')
    test.skip(configRes.status() !== 200, `Warp's settings are not available on this server (GET /api/warp/config answered ${configRes.status()})`)
    const config = (await configRes.json()) as { vector_store: string }
    const current = config.vector_store || stores.find((s) => s.primary)?.name || ''
    const original = stores.find((s) => s.name === current)
    const target = stores.find((s) => s.name !== current && s.status === 'connected')
    test.skip(
      original?.status !== 'connected' || !target,
      'needs Warp on a connected store and another connected store to switch to, so the test can switch back',
    )
    const from = original as ListedVectorStore
    const to = target as ListedVectorStore

    const pick = async (store: ListedVectorStore) => {
      await page.getByTestId('warp-vector-store-select').click()
      await page.getByTestId(`warp-vector-store-option-${store.name}`).click()
    }
    const save = async (store: ListedVectorStore) => {
      const put = page.waitForResponse((r) => new URL(r.url()).pathname.endsWith('/api/warp/config') && r.request().method() === 'PUT')
      await page.getByTestId('warp-save-btn').click()
      const res = await put
      expect(res.status(), await res.text()).toBe(200)
      expect(res.request().postDataJSON().vector_store).toBe(store.primary ? '' : store.name)
      await expect(page.getByTestId('warp-vector-store-status')).toContainText(store.name)
    }

    await openWarpSettings(page)
    await page.getByTestId('warp-vector-store-select').click()
    for (const store of stores) {
      await expect(page.getByTestId(`warp-vector-store-option-${store.name}`)).toBeVisible()
    }
    await page.keyboard.press('Escape')

    try {
      await pick(to)
      await expect(page.getByTestId('warp-vector-store-switch-note')).toBeVisible()
      await save(to)
      const after = (await (await page.request.get('/api/warp/config')).json()) as { connected_vector_store: { name: string } | null }
      expect(after.connected_vector_store?.name).toBe(to.name)
    } finally {
      // Put Warp back on the store it started on, if the switch was saved.
      const now = (await (await page.request.get('/api/warp/config')).json()) as { vector_store: string }
      if ((now.vector_store || stores.find((s) => s.primary)?.name) !== from.name) {
        await openWarpSettings(page)
        await pick(from)
        await save(from)
      }
    }
  })
})
