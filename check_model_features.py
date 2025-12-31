"""
Check what features the saved model expects
"""

from ml_training_pipeline import MLModelTrainer

model_path = 'walkforward_2022_jan_apr/models/Jan_2022.pkl'
print(f"Loading model: {model_path}")

trainer = MLModelTrainer.load_model(str(model_path))

print(f"\nModel expects {trainer.scaler.n_features_in_} features")

# Check if model metadata has feature names
if hasattr(trainer, 'feature_names'):
    print(f"\nFeature names ({len(trainer.feature_names)}):")
    for i, name in enumerate(trainer.feature_names, 1):
        print(f"  {i:3d}. {name}")
else:
    print("\nNo feature names saved in model metadata")

# Check the model itself
if hasattr(trainer.model, 'feature_names_in_'):
    print(f"\nModel feature names ({len(trainer.model.feature_names_in_)}):")
    for i, name in enumerate(trainer.model.feature_names_in_[:20], 1):  # First 20
        print(f"  {i:3d}. {name}")
    print(f"  ... and {len(trainer.model.feature_names_in_) - 20} more")
