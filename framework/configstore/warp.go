package configstore

import (
	"context"
	"errors"

	"github.com/maximhq/bifrost/framework/configstore/tables"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// WarpStore is a narrow optional interface, following NotificationStore: adding
// Warp must not force every ConfigStore test double in the tree to grow methods
// it will never implement.
type WarpStore interface {
	// GetWarpConfig returns the singleton config row, or (nil, nil) when Warp
	// has never been configured. A missing row is an expected state, not an
	// error — most deployments never turn Warp on.
	GetWarpConfig(ctx context.Context) (*tables.TableWarpConfig, error)
	// UpsertWarpConfig writes the singleton row.
	UpsertWarpConfig(ctx context.Context, config *tables.TableWarpConfig) error
}

// GetWarpConfig reads the singleton Warp row. A missing row returns (nil, nil):
// most deployments never turn Warp on, so absence is an ordinary state that
// callers should not have to distinguish from a failure at every call site.
func (s *RDBConfigStore) GetWarpConfig(ctx context.Context) (*tables.TableWarpConfig, error) {
	var config tables.TableWarpConfig
	err := s.DB().WithContext(ctx).First(&config, tables.WarpConfigRowID).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &config, nil
}

// UpsertWarpConfig writes the singleton Warp row, pinning the primary key so a
// caller that left ID unset cannot insert a second, autoincremented row that
// GetWarpConfig would never find.
func (s *RDBConfigStore) UpsertWarpConfig(ctx context.Context, config *tables.TableWarpConfig) error {
	// Pin the ID rather than trusting the caller: the table is a singleton by
	// contract, and a caller that passed 0 would otherwise have GORM insert a
	// second, autoincremented row that the read path above would never find.
	config.ID = tables.WarpConfigRowID
	// An explicit ON CONFLICT upsert rather than Save. Save's behaviour for a
	// non-zero primary key with no matching row differs by GORM version and
	// dialect - it can issue an UPDATE that quietly affects zero rows - and
	// "the settings saved but nothing changed" is a bug nobody can see.
	return s.DB().WithContext(ctx).Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "id"}},
		UpdateAll: true,
	}).Create(config).Error
}

// UpsertWarpConfigMerging writes the singleton row like UpsertWarpConfig, after calling merge
// with the row as stored now (nil when there is none), read under a lock held until the write
// commits. A value the write derives from the stored row, such as the indexes Warp's index left
// behind when it moved, is then computed from the row it replaces rather than from an earlier
// read, so two overlapping saves cannot drop what the other one recorded. merge may run twice
// and must compute what it sets from what it is given each time.
func (s *RDBConfigStore) UpsertWarpConfigMerging(ctx context.Context, config *tables.TableWarpConfig, merge func(stored *tables.TableWarpConfig) error) error {
	config.ID = tables.WarpConfigRowID
	err := s.upsertWarpConfigMergingOnce(ctx, config, merge)
	if errors.Is(err, errWarpRowTaken) {
		// Another first save inserted the row since this one found none: lock it and merge again.
		err = s.upsertWarpConfigMergingOnce(ctx, config, merge)
	}
	return err
}

// errWarpRowTaken is a first save finding that another one inserted the row first.
var errWarpRowTaken = errors.New("warp config row was inserted by another save")

func (s *RDBConfigStore) upsertWarpConfigMergingOnce(ctx context.Context, config *tables.TableWarpConfig, merge func(stored *tables.TableWarpConfig) error) error {
	return s.DB().WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var stored tables.TableWarpConfig
		err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&stored, tables.WarpConfigRowID).Error
		switch {
		case errors.Is(err, gorm.ErrRecordNotFound):
			// No row to lock, so two first saves can both get here. Each inserts without
			// overwriting; the one that finds the row taken starts over, and then merges from
			// the row the other wrote rather than from none.
			if err := merge(nil); err != nil {
				return err
			}
			result := tx.Clauses(clause.OnConflict{Columns: []clause.Column{{Name: "id"}}, DoNothing: true}).Create(config)
			if result.Error != nil {
				return result.Error
			}
			if result.RowsAffected == 0 {
				return errWarpRowTaken
			}
			return nil
		case err != nil:
			return err
		}
		if err := merge(&stored); err != nil {
			return err
		}
		return tx.Clauses(clause.OnConflict{Columns: []clause.Column{{Name: "id"}}, UpdateAll: true}).Create(config).Error
	})
}
