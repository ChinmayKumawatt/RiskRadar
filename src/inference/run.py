import argparse
import json

from src.inference.pipeline.inference_pipeline import InferencePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run RiskRadar inference")
    parser.add_argument(
        "--disease",
        required=False,
        help="Disease to predict, for example: heart, diabetes, ckd, hypertension",
    )
    parser.add_argument(
        "--payload",
        required=False,
        help='JSON payload string, for example: \'{"age": 63, "sex": 1}\'',
    )
    parser.add_argument(
        "--input-file",
        required=False,
        help="Path to a JSON file containing one record or a list of records",
    )
    parser.add_argument(
        "--list-diseases",
        action="store_true",
        help="List diseases that currently have complete inference artifacts",
    )
    return parser.parse_args()


def load_payload(args):
    if args.payload:
        return json.loads(args.payload)

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    raise ValueError("Provide either --payload or --input-file.")


if __name__ == "__main__":
    arguments = parse_args()
    pipeline = InferencePipeline()

    if arguments.list_diseases:
        print(pipeline.get_supported_diseases())
    else:
        if not arguments.disease:
            raise ValueError("--disease is required unless --list-diseases is used.")

        predictions = pipeline.predict(
            disease_name=arguments.disease,
            payload=load_payload(arguments),
        )
        print(json.dumps(predictions, indent=4))
