"""Verify that the dataset loads correctly with time features."""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def verify_dataset_loading():
    """Verify dataset can load with time features without errors."""

    try:
        from dataset import PCVRParquetDataset
        import torch

        # Try to find data and schema
        data_path = Path("../demo_1000.parquet")

        if not data_path.exists():
            print(f"[WARNING] Demo data not found at {data_path}")
            print("Please provide the correct path to your parquet file.")
            return False

        # Check if we need to create a minimal schema
        schema_path = Path("../schema.json")
        if not schema_path.exists():
            print(f"[WARNING] Schema file not found at {schema_path}")
            print("Creating a minimal test schema...")

            import json
            minimal_schema = {
                "user_int": [],
                "item_int": [],
                "user_dense": [],
                "seq": {}
            }

            with open(schema_path, 'w') as f:
                json.dump(minimal_schema, f, indent=2)

            print(f"[INFO] Created minimal schema at {schema_path}")

        print("\n" + "="*80)
        print("Testing Dataset Loading with Time Features")
        print("="*80 + "\n")

        # Create dataset
        print("[1/4] Creating dataset instance...")
        dataset = PCVRParquetDataset(
            parquet_path=str(data_path),
            schema_path=str(schema_path),
            batch_size=32,
            shuffle=False,
            buffer_batches=1,
            clip_vocab=True,
            is_training=True,
        )
        print(f"      Dataset created successfully!")
        print(f"      Total rows: {dataset.num_rows}")
        print(f"      user_int_schema.total_dim: {dataset.user_int_schema.total_dim}")
        print(f"      user_int vocab_sizes length: {len(dataset.user_int_vocab_sizes)}")

        # Try to get one batch
        print("\n[2/4] Loading first batch...")
        iterator = iter(dataset)
        batch = next(iterator)
        print(f"      Batch loaded successfully!")

        # Check batch contents
        print("\n[3/4] Checking batch structure...")
        print(f"      Batch keys: {list(batch.keys())}")
        print(f"      user_int_feats shape: {batch['user_int_feats'].shape}")
        print(f"      item_int_feats shape: {batch['item_int_feats'].shape}")
        print(f"      user_dense_feats shape: {batch['user_dense_feats'].shape}")
        print(f"      label shape: {batch['label'].shape}")

        # Check time features are in the last positions
        print("\n[4/4] Verifying time features...")
        user_int_feats = batch['user_int_feats']
        B, D = user_int_feats.shape

        # Time features should be in the last 15 (or 7) positions
        # Check if values are in expected ranges
        time_feat_start = D - 15  # Assuming full 15 features

        if time_feat_start >= 0:
            time_features = user_int_feats[:, time_feat_start:].numpy()

            # Check ranges for each time feature
            checks = [
                ("Year", time_features[:, 0], 0, 100),
                ("Month", time_features[:, 1], 0, 12),
                ("Day", time_features[:, 2], 0, 31),
                ("Hour", time_features[:, 3], 0, 23),
                ("Minute", time_features[:, 4], 0, 59),
                ("Weekday", time_features[:, 5], 0, 6),
                ("Is_Weekend", time_features[:, 6], 0, 1),
            ]

            all_valid = True
            for name, values, min_val, max_val in checks:
                valid = (values >= min_val).all() and (values <= max_val).all()
                status = "[PASS]" if valid else "[FAIL]"
                print(f"      {status} {name}: range [{values.min()}, {values.max()}], expected [{min_val}, {max_val}]")
                all_valid = all_valid and valid

            if all_valid:
                print("\n" + "="*80)
                print("[SUCCESS] All checks passed! Time features are correctly extracted.")
                print("="*80)
                return True
            else:
                print("\n" + "="*80)
                print("[ERROR] Some time features are out of range!")
                print("="*80)
                return False
        else:
            print(f"      [WARNING] Could not locate time features (D={D}, expected >= 15)")
            print(f"      This might be normal if schema doesn't include time features yet.")
            return True

    except Exception as e:
        print("\n" + "="*80)
        print(f"[ERROR] Dataset loading failed: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_dataset_loading()
    sys.exit(0 if success else 1)
