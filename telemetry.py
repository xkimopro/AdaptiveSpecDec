import jsonlines
import time

class Telemetry:
    """
    Logs telemetry for speculative decoding steps to a JSONL file.
    """
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.writer = jsonlines.open(output_path, mode='w')

    def log_step(self, data: dict):
        """
        Logs a single step's telemetry data.
        """
        data['timestamp'] = time.time()
        self.writer.write(data)

    def close(self):
        """
        Closes the JSONL writer.
        """
        self.writer.close()

if __name__ == '__main__':
    # Example of how to use the Telemetry class
    telemetry = Telemetry("test_telemetry.jsonl")
    telemetry.log_step({
        "features": {
            "prompt_length": 50,
            "entropy": 2.3,
            "logit_gap": 0.1,
            "kl_divergence": 0.05,
            "position_index": 10
        },
        "results": {
            "accept_mask": [True, True, False, True],
            "accepted_prefix_length": 2,
            "runtimes": {"draft": 0.1, "verify": 0.05}
        }
    })
    telemetry.close()
    print("Wrote test telemetry to test_telemetry.jsonl")
