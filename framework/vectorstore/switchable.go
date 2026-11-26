package vectorstore

import (
	"context"
	"errors"
	"sync/atomic"
)

// ErrNoStore is what a Switchable answers while it holds no store.
var ErrNoStore = errors.New("vector store is not available")

// Switchable is a VectorStore that forwards every call to the store it holds now, which Swap
// replaces atomically. A consumer whose store can be changed at runtime keeps one Switchable and
// hands it to everything that reads or writes vectors, so they all follow a change without being
// rebuilt. A call already forwarded finishes on the store it reached. While it holds no store,
// every call fails with ErrNoStore.
type Switchable struct {
	current atomic.Pointer[heldStore]
}

// heldStore boxes the interface, since atomic.Pointer holds a pointer.
type heldStore struct{ store VectorStore }

// NewSwitchable returns a Switchable holding store, which may be nil.
func NewSwitchable(store VectorStore) *Switchable {
	s := &Switchable{}
	s.Swap(store)
	return s
}

// Swap makes store the one every later call reaches, and returns the one it replaces.
func (s *Switchable) Swap(store VectorStore) (previous VectorStore) {
	if old := s.current.Swap(&heldStore{store: store}); old != nil {
		return old.store
	}
	return nil
}

// Current returns the store calls reach now, or nil.
func (s *Switchable) Current() VectorStore {
	if held := s.current.Load(); held != nil {
		return held.store
	}
	return nil
}

func (s *Switchable) target() (VectorStore, error) {
	store := s.Current()
	if store == nil {
		return nil, ErrNoStore
	}
	return store, nil
}

// FiltersVectorReadsOnServer forwards ServerFilteredReader to the store held now.
func (s *Switchable) FiltersVectorReadsOnServer() bool {
	store := s.Current()
	return store != nil && FiltersVectorReadsOnServer(store)
}

func (s *Switchable) Ping(ctx context.Context) error {
	store, err := s.target()
	if err != nil {
		return err
	}
	return store.Ping(ctx)
}

func (s *Switchable) CreateNamespace(ctx context.Context, namespace string, dimension int, properties map[string]VectorStoreProperties) error {
	store, err := s.target()
	if err != nil {
		return err
	}
	return store.CreateNamespace(ctx, namespace, dimension, properties)
}

func (s *Switchable) DeleteNamespace(ctx context.Context, namespace string) error {
	store, err := s.target()
	if err != nil {
		return err
	}
	return store.DeleteNamespace(ctx, namespace)
}

func (s *Switchable) ListNamespaces(ctx context.Context, prefix string) ([]string, error) {
	store, err := s.target()
	if err != nil {
		return nil, err
	}
	return store.ListNamespaces(ctx, prefix)
}

func (s *Switchable) GetChunk(ctx context.Context, namespace string, id string) (SearchResult, error) {
	store, err := s.target()
	if err != nil {
		return SearchResult{}, err
	}
	return store.GetChunk(ctx, namespace, id)
}

func (s *Switchable) GetChunks(ctx context.Context, namespace string, ids []string) ([]SearchResult, error) {
	store, err := s.target()
	if err != nil {
		return nil, err
	}
	return store.GetChunks(ctx, namespace, ids)
}

func (s *Switchable) GetAll(ctx context.Context, namespace string, queries []Query, selectFields []string, cursor *string, limit int64) ([]SearchResult, *string, error) {
	store, err := s.target()
	if err != nil {
		return nil, nil, err
	}
	return store.GetAll(ctx, namespace, queries, selectFields, cursor, limit)
}

func (s *Switchable) GetNearest(ctx context.Context, namespace string, vector []float32, queries []Query, selectFields []string, threshold float64, limit int64) ([]SearchResult, error) {
	store, err := s.target()
	if err != nil {
		return nil, err
	}
	return store.GetNearest(ctx, namespace, vector, queries, selectFields, threshold, limit)
}

// RequiresVectors answers for the store held now; with none, false.
func (s *Switchable) RequiresVectors() bool {
	store := s.Current()
	return store != nil && store.RequiresVectors()
}

func (s *Switchable) Add(ctx context.Context, namespace string, id string, embedding []float32, metadata map[string]interface{}) error {
	store, err := s.target()
	if err != nil {
		return err
	}
	return store.Add(ctx, namespace, id, embedding, metadata)
}

func (s *Switchable) Delete(ctx context.Context, namespace string, id string) error {
	store, err := s.target()
	if err != nil {
		return err
	}
	return store.Delete(ctx, namespace, id)
}

func (s *Switchable) DeleteAll(ctx context.Context, namespace string, queries []Query) ([]DeleteResult, error) {
	store, err := s.target()
	if err != nil {
		return nil, err
	}
	return store.DeleteAll(ctx, namespace, queries)
}

func (s *Switchable) CountVectors(ctx context.Context, namespace string, queries []Query) (int64, error) {
	store, err := s.target()
	if err != nil {
		return 0, err
	}
	return store.CountVectors(ctx, namespace, queries)
}

// Close does nothing: the stores a Switchable forwards to are owned elsewhere (the Registry),
// and closing one through it would close it for everything else that uses it.
func (s *Switchable) Close(context.Context, string) error { return nil }
