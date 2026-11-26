package vectorstore

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// serverFiltered is a store that filters paged reads itself.
type serverFiltered struct{ VectorStore }

func (serverFiltered) FiltersVectorReadsOnServer() bool { return true }

func memoryStore(t *testing.T) VectorStore {
	t.Helper()
	store, err := NewVectorStore(context.Background(), &Config{Enabled: true, Type: VectorStoreTypeChromem}, nil)
	require.NoError(t, err)
	return store
}

// Every call reaches the store held now: a swap moves what follows to the new store and leaves
// the old one as it was. With no store, every call fails with ErrNoStore.
func TestSwitchableForwardsToTheStoreHeldNow(t *testing.T) {
	ctx := context.Background()
	first, second := memoryStore(t), memoryStore(t)
	s := NewSwitchable(first)
	require.NoError(t, s.CreateNamespace(ctx, "logs", 2, nil))
	require.NoError(t, s.Add(ctx, "logs", "a", []float32{1, 0}, nil))

	assert.Same(t, first, s.Swap(second), "Swap returns the store it replaces")
	assert.Same(t, second, s.Current())
	_, err := s.GetChunk(ctx, "logs", "a")
	assert.Error(t, err, "the new store does not have what the old one holds")
	require.NoError(t, s.CreateNamespace(ctx, "logs", 2, nil))
	require.NoError(t, s.Add(ctx, "logs", "b", []float32{0, 1}, nil))
	count, err := first.CountVectors(ctx, "logs", nil)
	require.NoError(t, err)
	assert.EqualValues(t, 1, count, "the old store is left as it was")

	s.Swap(nil)
	assert.Nil(t, s.Current())
	assert.ErrorIs(t, s.Add(ctx, "logs", "c", []float32{1, 1}, nil), ErrNoStore)
	_, err = s.CountVectors(ctx, "logs", nil)
	assert.ErrorIs(t, err, ErrNoStore)
	assert.False(t, s.RequiresVectors())
	assert.ErrorIs(t, NewSwitchable(nil).Ping(ctx), ErrNoStore)
}

// The optional reader interface follows the store held now, and closing a Switchable leaves its
// store open for everything else that uses it.
func TestSwitchableForwardsServerFilteringAndKeepsStoresOpen(t *testing.T) {
	ctx := context.Background()
	plain := memoryStore(t)
	s := NewSwitchable(plain)
	assert.False(t, FiltersVectorReadsOnServer(s))
	s.Swap(serverFiltered{plain})
	assert.True(t, FiltersVectorReadsOnServer(s))

	require.NoError(t, s.Close(ctx, ""))
	require.NoError(t, s.CreateNamespace(ctx, "still-open", 2, nil), "the store is not closed through the Switchable")
}

// Swaps and calls may run at once; run with -race.
func TestSwitchableSwapsUnderConcurrentCalls(t *testing.T) {
	ctx := context.Background()
	stores := []VectorStore{memoryStore(t), memoryStore(t)}
	s := NewSwitchable(stores[0])
	var wg sync.WaitGroup
	for i := range 4 {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for range 100 {
				s.Swap(stores[i%2])
			}
		}()
		go func() {
			defer wg.Done()
			for range 100 {
				_, _ = s.CountVectors(ctx, "logs", nil)
			}
		}()
	}
	wg.Wait()
}
