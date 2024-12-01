import os
import ezkl
import json
import numpy as np
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate proof for a given model and data.")
    parser.add_argument('--output', type=str, required=True, help='Output folder path')
    parser.add_argument('--data', type=str, required=True, help='Data file path')
    parser.add_argument('--model', type=str, required=True, help='Model file path')
    parser.add_argument('--mode', type=str, required=False, help='Model file path')

    args = parser.parse_args()
    output_folder = args.output

    compiled_model_path = os.path.join(output_folder, 'network.compiled')
    settings_path = os.path.join(output_folder, 'settings.json') 
    witness_path = os.path.join(output_folder, 'witness.json')

    proof_path = os.path.join(output_folder, 'proof.json')

    pk_path = os.path.join(output_folder, 'test.pk')
    vk_path = os.path.join(output_folder, 'test.vk')

    # try to run ezkl.prove
    # proof = ezkl.prove(
    #     witness_path,
    #     compiled_model_path,
    #     pk_path,
    #     proof_path,
    #     "single",
    # )
    try:
        start_time = time.time()
        res = ezkl.prove(witness_path, compiled_model_path, pk_path, proof_path, "single")
        print(f"Proof generation time: {time.time() - start_time}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    assert os.path.isfile(proof_path)

    # run ezkl command instead python binging (ezkl.prove)
    # command = f"ezkl prove --witness {witness_path} --compiled-circuit {compiled_model_path} --pk-path {pk_path} --proof-path {proof_path}"
    # # try to run the command and catch the exception
    # try:
    #     os.system(command)
    # except Exception as e:
    #     print(f"Error: {e}")
    #     exit(1)


    print ('Proof Gen')