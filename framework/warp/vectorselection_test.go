package warp

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/maximhq/bifrost/framework/logstore"
	"github.com/maximhq/bifrost/framework/vectorstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Reads that choose different stores at once leave the choice and the store
// calls reach in step: whichever read lands last, the store in use is the one
// the choice names.
func TestWarpVectorStoreChoiceAndStoreChangeTogether(t *testing.T) {
	stores, _, _ := warpStoresForTest()
	service := serviceForTest(t, nil, WithConfigStore(&recordingStore{row: validWarpConfigRow()}), WithVectorStores(stores))
	rows := []*tables.TableWarpConfig{{VectorStore: ""}, {VectorStore: "warp-qdrant"}}
	for round := range 50 {
		var wg sync.WaitGroup
		for i := range 8 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				service.followVectorStore(rows[(i+round)%2])
			}()
		}
		wg.Wait()
		selection := service.selection.Load()
		require.Same(t, selection.store(), service.currentVectorStore(), "round %d: the choice names the store in use", round)
	}
}

// While the store the index is in is configured but cannot be reached (it
// failed at startup), a deleted log's delete is queued, and a retry keeps it
// until the store can be reached. A store no longer configured at all cannot
// be reached by any later pass, so nothing is queued for it.
func TestWarpDeletesWaitForAnUnreachableStoreTheIndexIsIn(t *testing.T) {
	ctx := context.Background()
	stores, _, _ := warpStoresForTest()
	row := validWarpConfigRow()
	row.VectorStore = "down"
	service := serviceForTest(t, nil, WithConfigStore(&recordingStore{row: row}), WithVectorStores(stores),
		WithLogReader(topicTestReader(t, map[string]logstore.Log{})))

	service.ForgetLogs(ctx, []string{"gone"})
	assert.Equal(t, map[string]string{"gone": schemas.WarpDefaultLogVectorStoreNamespace}, pendingVectorDeletes(t, service))
	done, err := service.retryVectorDeletes(ctx)
	require.NoError(t, err)
	assert.Zero(t, done)
	assert.Contains(t, pendingVectorDeletes(t, service), "gone", "still queued while the store is down")

	row.VectorStore = "no-longer-configured"
	service.ForgetLogs(ctx, []string{"other"})
	assert.NotContains(t, pendingVectorDeletes(t, service), "other")
}

// When the configuration cannot be read, which store holds the vectors is not
// known, so every delete is queued for the retry, even when the primary store
// (the only one known without the configuration) is down.
func TestWarpDeletesAreQueuedWhenTheConfigurationCannotBeRead(t *testing.T) {
	ctx := context.Background()
	named := newFakeWarpVectorStore()
	stores := &fakeVectorStores{
		primary: &vectorstore.NamedStore{Name: "shared", Type: "weaviate", ID: "weaviate+http://shared", Primary: true, Status: vectorstore.StoreFailed, Err: errors.New("refused")},
		named:   []*vectorstore.NamedStore{{Name: "warp-qdrant", Type: "qdrant", ID: "qdrant://warp:6334", Status: vectorstore.StoreConnected, Store: named}},
	}
	row := validWarpConfigRow()
	row.VectorStore = "warp-qdrant"
	config := &flakyConfigStore{recordingStore: &recordingStore{row: row}, err: errors.New("database is closed")}
	service := serviceForTest(t, nil, WithConfigStore(config), WithVectorStores(stores),
		WithLogReader(topicTestReader(t, map[string]logstore.Log{})))

	service.ForgetLogs(ctx, []string{"a", "b"})
	pending := pendingVectorDeletes(t, service)
	assert.Contains(t, pending, "a")
	assert.Contains(t, pending, "b")
	assert.Empty(t, pending["a"], "the namespace is read when the retry runs")
}

// A retried delete recorded without a namespace takes it from the same read of
// the configuration as the store, so the two are always one configuration's.
func TestWarpRetriedDeletesTakeTheStoreAndNamespaceFromOneRead(t *testing.T) {
	ctx := context.Background()
	stores, shared, qdrant := warpStoresForTest()
	row := validWarpConfigRow()
	row.VectorStore = "warp-qdrant"
	row.LogVectorStoreNamespace = "QdrantWarpLogs"
	reads := &countingReads{recordingStore: &recordingStore{row: row}}
	service := serviceForTest(t, nil, WithConfigStore(reads), WithVectorStores(stores),
		WithLogReader(topicTestReader(t, map[string]logstore.Log{})))
	require.NoError(t, qdrant.Add(ctx, "QdrantWarpLogs", "queued", []float32{1, 0}, map[string]interface{}{"warp_log": true}))
	require.NoError(t, service.topicStore().AddWarpVectorDeletes(ctx, []logstore.WarpVectorDelete{{LogID: "queued"}}))

	done, err := service.retryVectorDeletes(ctx)
	require.NoError(t, err)
	assert.Equal(t, 1, done)
	assert.Empty(t, qdrant.idsIn("QdrantWarpLogs"))
	assert.Empty(t, shared.idsIn("QdrantWarpLogs"))
	assert.Equal(t, 1, reads.reads, "one read of the configuration")
}

// countingReads counts reads of the configuration.
type countingReads struct {
	*recordingStore
	reads int
}

func (c *countingReads) GetWarpConfig(ctx context.Context) (*tables.TableWarpConfig, error) {
	c.reads++
	return c.recordingStore.GetWarpConfig(ctx)
}

// movedServiceForTest is a service whose index was in the primary store "shared",
// holding Warp's vectors, and has just moved to "warp-qdrant".
func movedServiceForTest(t *testing.T) (*Service, *recordingStore, *fakeWarpVectorStore, *fakeWarpVectorStore) {
	t.Helper()
	ctx := context.Background()
	stores, shared, qdrant := warpStoresForTest()
	store := &recordingStore{}
	service := serviceForTest(t, nil, WithConfigStore(store), WithVectorStores(stores),
		WithLogReader(topicTestReader(t, map[string]logstore.Log{})))
	_, err := service.SaveConfig(ctx, validWarpConfigInput())
	require.NoError(t, err)
	shared.existing = []string{schemas.WarpDefaultLogVectorStoreNamespace}
	require.NoError(t, shared.Add(ctx, schemas.WarpDefaultLogVectorStoreNamespace, "old", []float32{1, 0}, map[string]interface{}{"warp_log": true}))
	moved := validWarpConfigInput()
	moved.VectorStore = "warp-qdrant"
	_, err = service.SaveConfig(ctx, moved)
	require.NoError(t, err)
	return service, store, shared, qdrant
}

// The index Warp moved away from is deleted by upkeep once the grace for writes
// already in flight has passed, and then taken off the list.
func TestWarpDeletesTheIndexItMovedAwayFromAfterTheGrace(t *testing.T) {
	ctx := context.Background()
	service, store, shared, _ := movedServiceForTest(t)
	left := pendingCleanups(store.row)
	require.Len(t, left, 1)

	deleted, err := service.deleteLeftIndexes(ctx, left[0].Since.Add(vectorStoreCleanupGrace/2))
	require.NoError(t, err)
	assert.Zero(t, deleted, "kept while writes in flight may still land")
	assert.Empty(t, shared.deleted)

	deleted, err = service.deleteLeftIndexes(ctx, left[0].Since.Add(vectorStoreCleanupGrace))
	require.NoError(t, err)
	assert.Equal(t, 1, deleted)
	assert.Equal(t, []string{schemas.WarpDefaultLogVectorStoreNamespace}, shared.deleted, "a namespace holding only Warp's vectors goes whole")
	assert.Empty(t, pendingCleanups(store.row))
	assert.NotEmpty(t, store.row.VectorStore, "nothing else in the row changes")
	assert.True(t, store.row.Enabled)
}

// Upkeep runs the deletion on its timer pass.
func TestWarpUpkeepDeletesTheIndexItMovedAwayFrom(t *testing.T) {
	service, store, shared, _ := movedServiceForTest(t)
	cleanup := pendingCleanups(store.row)
	cleanup[0].Since = time.Now().Add(-time.Hour)
	encoded, err := encodeCleanups(cleanup)
	require.NoError(t, err)
	store.row.VectorStoreCleanup = encoded
	service.topicUpkeepTick(context.Background())
	assert.Equal(t, []string{schemas.WarpDefaultLogVectorStoreNamespace}, shared.deleted)
	assert.Empty(t, pendingCleanups(store.row))
}

// A namespace something besides Warp writes to keeps what is not Warp's.
func TestWarpDeletesOnlyItsOwnVectorsFromASharedNamespace(t *testing.T) {
	ctx := context.Background()
	service, store, shared, _ := movedServiceForTest(t)
	require.NoError(t, shared.Add(ctx, schemas.WarpDefaultLogVectorStoreNamespace, "not-warps", []float32{0, 1}, map[string]interface{}{"warp_log": false}))
	deleted, err := service.deleteLeftIndexes(ctx, time.Now().Add(time.Hour))
	require.NoError(t, err)
	assert.Equal(t, 1, deleted)
	assert.Empty(t, shared.deleted, "the namespace stays")
	assert.Equal(t, []string{"not-warps"}, shared.idsIn(schemas.WarpDefaultLogVectorStoreNamespace))
	assert.Empty(t, pendingCleanups(store.row))
}

// An index left in a store that cannot be reached now waits for it; one left in a
// store no longer configured waits for it to be configured again, unless the store
// kept everything in memory, which went with the process. A store renamed in
// config.json since is found by where it keeps its data.
func TestWarpKeepsALeftIndexUntilItsStoreCanBeReached(t *testing.T) {
	ctx := context.Background()
	stores, _, qdrant := warpStoresForTest()
	qdrant.existing = []string{"QdrantWarpLogs"}
	since := time.Now().Add(-time.Hour)
	left := []vectorStoreCleanup{
		{Store: "down", StoreID: "redis://down:6379/0", Namespace: "A", Dimension: 2, Since: since},
		{Store: "gone", StoreID: "qdrant://gone:6334", Namespace: "B", Dimension: 2, Since: since},
		{Store: "memory", StoreID: "chromem#0011", InMemory: true, Namespace: "C", Dimension: 2, Since: since},
		{Store: "old-name", StoreID: "qdrant://warp:6334", Namespace: "QdrantWarpLogs", Dimension: 2, Since: since},
	}
	encoded, err := encodeCleanups(left)
	require.NoError(t, err)
	row := validWarpConfigRow()
	row.VectorStoreCleanup = encoded
	store := &recordingStore{row: row}
	service := serviceForTest(t, nil, WithConfigStore(store), WithVectorStores(stores))

	deleted, err := service.deleteLeftIndexes(ctx, time.Now())
	require.NoError(t, err)
	assert.Equal(t, 2, deleted)
	assert.Equal(t, []string{"QdrantWarpLogs"}, qdrant.deleted, "found by its data under its new name")
	var remaining []string
	for _, entry := range pendingCleanups(store.row) {
		remaining = append(remaining, entry.Store)
	}
	assert.Equal(t, []string{"down", "gone"}, remaining)
}

// A persistent Chromem store the index moved away from is opened again after a
// restart to delete Warp's index there, which nothing indexed there does any more.
func TestWarpDeletesTheIndexItMovedAwayFromAfterARestart(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "old-chromem")
	open := func() vectorstore.VectorStore {
		t.Helper()
		store, err := vectorstore.NewVectorStore(ctx, &vectorstore.Config{Enabled: true, Type: vectorstore.VectorStoreTypeChromem, Config: vectorstore.ChromemConfig{Path: path}}, nil)
		require.NoError(t, err)
		return store
	}
	namespace := "OldWarpLogs"
	before := open()
	_, err := ensureWarpNamespace(ctx, before, namespace, 3)
	require.NoError(t, err)
	require.NoError(t, before.Add(ctx, namespace, "stale", []float32{1, 0, 0}, map[string]interface{}{"warp_log": true}))
	require.NoError(t, before.Close(ctx, ""))

	restarted := open() // the same directory, opened by a new process: no namespace is open yet
	stores, _, _ := warpStoresForTest()
	stores.named = append(stores.named, &vectorstore.NamedStore{Name: "old-chromem", Type: vectorstore.VectorStoreTypeChromem, ID: "chromem://" + path, Status: vectorstore.StoreConnected, Store: restarted})
	encoded, err := encodeCleanups([]vectorStoreCleanup{{Store: "old-chromem", StoreID: "chromem://" + path, Namespace: namespace, Dimension: 3, Since: time.Now().Add(-time.Hour)}})
	require.NoError(t, err)
	row := validWarpConfigRow()
	row.VectorStoreCleanup = encoded
	service := serviceForTest(t, nil, WithConfigStore(&recordingStore{row: row}), WithVectorStores(stores))

	deleted, err := service.deleteLeftIndexes(ctx, time.Now())
	require.NoError(t, err)
	assert.Equal(t, 1, deleted)
	namespaces, err := restarted.ListNamespaces(ctx, namespace)
	require.NoError(t, err)
	assert.Empty(t, namespaces)
}

// Moving back into a store the index left before upkeep deleted it deletes that
// index first, so the new one starts empty: the log deletes made meanwhile only
// reached the store the index was in. Each move starts a new index.
func TestWarpMovingBackDeletesTheIndexItLeftThere(t *testing.T) {
	ctx := context.Background()
	service, store, shared, _ := movedServiceForTest(t)
	firstMove := store.row.VectorIndexID
	back := inputFromJSON(t, `{"enabled": true, "provider": "openai", "model": "gpt-4o", "vector_store": "", "embedding_provider": "openai", "embedding_model": "text-embedding-3-small", "embedding_dimension": 1536}`)
	_, err := service.SaveConfig(ctx, back)
	require.NoError(t, err)
	assert.Equal(t, []string{schemas.WarpDefaultLogVectorStoreNamespace}, shared.deleted, "deleted before it is used again")
	assert.NotEqual(t, firstMove, store.row.VectorIndexID)
	assert.NotEmpty(t, store.row.VectorIndexID)
	left := pendingCleanups(store.row)
	require.Len(t, left, 1, "only the qdrant index is left")
	assert.Equal(t, "qdrant://warp:6334", left[0].StoreID)
}

// A save that read the settings before another save changed the store is
// refused: what it checked was for a store the index is no longer in, and an
// omitted choice would otherwise move the index back.
func TestWarpRefusesASaveThatReadBeforeAnotherSaveChangedTheStore(t *testing.T) {
	ctx := context.Background()
	stores, _, _ := warpStoresForTest()
	stored := validWarpConfigRow()
	stored.VectorStore, stored.VectorIndexID = "warp-qdrant", "moved"
	stale := validWarpConfigRow() // what this save read before that move committed
	store := &staleFirstRead{recordingStore: &recordingStore{row: stored}, stale: stale}
	service := serviceForTest(t, nil, WithConfigStore(store), WithVectorStores(stores))

	_, err := service.SaveConfig(ctx, inputFromJSON(t, `{"enabled": true, "provider": "openai", "model": "gpt-4o"}`))
	require.ErrorIs(t, err, ErrVectorStoreChoiceChanged)
	assert.Equal(t, "warp-qdrant", store.row.VectorStore, "the other save's choice stands")
	assert.Equal(t, "moved", store.row.VectorIndexID)

	_, err = service.SaveConfig(ctx, inputFromJSON(t, `{"enabled": true, "provider": "openai", "model": "gpt-4o"}`))
	require.NoError(t, err, "saved again after a fresh read")
	assert.Equal(t, "warp-qdrant", store.row.VectorStore)
}

// staleFirstRead serves one stale row to the next read, as a save that read
// before another save committed has seen, and the stored row after.
type staleFirstRead struct {
	*recordingStore
	stale *tables.TableWarpConfig
}

func (s *staleFirstRead) GetWarpConfig(ctx context.Context) (*tables.TableWarpConfig, error) {
	if stale := s.stale; stale != nil {
		s.stale = nil
		return stale, nil
	}
	return s.recordingStore.GetWarpConfig(ctx)
}

// A store renamed in config.json keeps the same data, so choosing it under its
// new name is not a move: the index and nothing to delete stay as they were.
func TestWarpChoosingARenamedStoreIsNotAMove(t *testing.T) {
	ctx := context.Background()
	stores, _, qdrant := warpStoresForTest()
	stores.named[0].Name = "renamed"
	row := validWarpConfigRow()
	row.VectorStore, row.VectorStoreID, row.VectorIndexID = "warp-qdrant", "qdrant://warp:6334", "built"
	store := &recordingStore{row: row}
	service := serviceForTest(t, nil, WithConfigStore(store), WithVectorStores(stores))

	renamed := validWarpConfigInput()
	renamed.VectorStore = "renamed"
	_, err := service.SaveConfig(ctx, renamed)
	require.NoError(t, err)
	assert.Equal(t, "renamed", store.row.VectorStore)
	assert.Equal(t, "built", store.row.VectorIndexID, "the same index")
	assert.Empty(t, pendingCleanups(store.row), "nothing left behind")
	assert.Same(t, qdrant, service.currentVectorStore())
}

// An index write goes to the store its configuration read named, with the
// namespace provisioned there: a move followed while the embedding call is in
// flight does not send the vector into the other store.
func TestWarpIndexWritesGoToTheStoreTheirConfigurationNamed(t *testing.T) {
	ctx := context.Background()
	stores, shared, qdrant := warpStoresForTest()
	row := assignTestRow()
	var service *Service
	moveDuringEmbedding := false
	embed := func(bctx *schemas.BifrostContext, request *schemas.BifrostEmbeddingRequest) (*schemas.BifrostEmbeddingResponse, *schemas.BifrostError) {
		if moveDuringEmbedding {
			service.followVectorStore(&tables.TableWarpConfig{VectorStore: "warp-qdrant"}) // another read saw the move
		}
		return assignTestEmbedding(bctx, request)
	}
	service = serviceForTest(t, nil, WithConfigStore(&recordingStore{row: row}), WithVectorStores(stores),
		WithLogReader(topicTestReader(t, map[string]logstore.Log{})), WithEmbeddingExecutor(embed))
	index := func(id string) {
		t.Helper()
		outcome, err := service.indexer.Index(ctx, &logstore.Log{
			ID: id, Timestamp: time.Now(), Object: string(schemas.ChatCompletionRequest), Status: "success",
			InputHistoryParsed: []schemas.ChatMessage{chatTurn(schemas.ChatMessageRoleUser, "how do I get a refund")},
		})
		require.NoError(t, err)
		require.Equal(t, IndexOutcomeIndexed, outcome)
	}

	moveDuringEmbedding = true
	index("in-flight")
	assert.Equal(t, []string{"in-flight"}, retentionTestIDs(shared), "written where its configuration read said")
	assert.Empty(t, retentionTestIDs(qdrant))

	moveDuringEmbedding = false
	row.VectorStore = "warp-qdrant"
	index("after")
	assert.Equal(t, []string{"after"}, retentionTestIDs(qdrant))
	assert.Equal(t, 1, qdrant.createCalls, "the namespace is provisioned in the store moved to, though it has the same name")
}

// The settings view shows the status of the store the row it returns names,
// whatever store this node holds at that moment.
func TestWarpConfigViewShowsTheStoreOfItsOwnRow(t *testing.T) {
	stores, _, _ := warpStoresForTest()
	service := serviceForTest(t, nil, WithConfigStore(&recordingStore{row: validWarpConfigRow()}), WithVectorStores(stores))
	service.followVectorStore(&tables.TableWarpConfig{VectorStore: "warp-qdrant"}) // another read, another row

	view := service.configViewFromRow(validWarpConfigRow())
	assert.Empty(t, view.VectorStore)
	require.NotNil(t, view.ConnectedVectorStore)
	assert.Equal(t, "shared", view.ConnectedVectorStore.Name)

	down := validWarpConfigRow()
	down.VectorStore = "down"
	view = service.configViewFromRow(down)
	assert.Nil(t, view.ConnectedVectorStore)
	assert.False(t, view.VectorStoreConnected)
	assert.Equal(t, `Warp's vector store "down" could not be reached at startup`, view.VectorStoreError)
}

// A save that moves the index is admitted under the backfill admission lock,
// so this process cannot admit a backfill for the store the index is leaving;
// a save that does not move it does not wait.
func TestWarpMoveWaitsForTheBackfillAdmissionLock(t *testing.T) {
	ctx := context.Background()
	stores, _, _ := warpStoresForTest()
	store := &recordingStore{row: validWarpConfigRow()}
	service := serviceForTest(t, nil, WithConfigStore(store), WithVectorStores(stores))
	unlock := service.LockBackfillAdmission()

	_, err := service.SaveConfig(ctx, inputFromJSON(t, `{"enabled": true, "provider": "openai", "model": "gpt-4o", "max_iterations": 4}`))
	require.NoError(t, err, "a save that does not move the index is not held back")

	saved := make(chan error, 1)
	go func() {
		moved := validWarpConfigInput()
		moved.VectorStore = "warp-qdrant"
		_, err := service.SaveConfig(ctx, moved)
		saved <- err
	}()
	select {
	case err := <-saved:
		t.Fatalf("the move was saved while a backfill start held the lock: %v", err)
	case <-time.After(100 * time.Millisecond):
	}
	unlock()
	require.NoError(t, <-saved)
	assert.Equal(t, "warp-qdrant", store.row.VectorStore)
}
