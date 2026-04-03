from src.training.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    results = pipeline.start()

    print("\nFinal Results:\n", results)