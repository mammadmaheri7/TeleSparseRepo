import asyncio
import os
import ezkl
import json
import numpy as np
import argparse
import time

async def gen_proof(output_folder, data_path , model_path, mode = "resources"):
    compiled_model_path = os.path.join(output_folder, 'network.compiled')
    settings_path = os.path.join(output_folder, 'settings.json') 
    witness_path = os.path.join(output_folder, 'witness.json')

    proof_path = os.path.join(output_folder, 'proof.json')

    pk_path = os.path.join(output_folder, 'test.pk')
    vk_path = os.path.join(output_folder, 'test.vk')


    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "public"
    # run_args.param_visibility = "fixed"
    run_args.param_visibility = "fixed"
    run_args.output_visibility = "public"

    print("Generating settings")
    try:
        res = ezkl.gen_settings(model_path, settings_path, py_run_args=run_args)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    assert res == True

    print("Calibrating settings")
    try:
        res = await ezkl.calibrate_settings(data_path, model_path, settings_path, mode, scales=[2,5,7,10], lookup_safety_margin=1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    assert res == True

    # print all settings file content
    with open(settings_path, "r") as f:
        setting = json.load(f)
        print(setting)
    
    print("Compiling circuit")
    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    assert res == True

    # srs path
    print("Getting SRS")
    res = await ezkl.get_srs(settings_path)
    assert res == True

    # now generate the witness file
    try:
        res = await ezkl.gen_witness(data_path, compiled_model_path, witness_path)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    assert os.path.isfile(witness_path)

    with open(witness_path, "r") as f:
        wit = json.load(f)

    with open(settings_path, "r") as f:
        setting = json.load(f)


    prediction_array = []

    # for value in wit['pretty_elements']['rescaled_outputs']:
    #     for field_element in value:
    #         # prediction_array.append(ezkl.vecu64_to_float(field_element, setting['model_output_scales'][0]))
    #         prediction_array.append(field_element)

    for value in wit["outputs"]:
        for field_element in value:
            prediction_array.append(ezkl.felt_to_float(field_element, setting['model_output_scales'][0]))

    pred = np.argmax([prediction_array])

    print("Mocking")
    res = ezkl.mock(witness_path, compiled_model_path)
    assert res == True

    print("Setting up")
    res = ezkl.setup(
            compiled_model_path,
            vk_path,
            pk_path,
        )
    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    # Generate the proof
    print("Generating proof")
    # proof = ezkl.prove(
    #         witness_path,
    #         compiled_model_path,
    #         pk_path,
    #         proof_path,
    #         "single",
    #     )
    command = f"ezkl prove --witness {witness_path} --compiled-circuit {compiled_model_path} --pk-path {pk_path} --proof-path {proof_path}"
    os.system(command)    
    #print(proof)
    assert os.path.isfile(proof_path)

    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate proof for a given model and data.")
    parser.add_argument('--output', type=str, required=True, help='Output folder path')
    parser.add_argument('--data', type=str, required=True, help='Data file path')
    parser.add_argument('--model', type=str, required=True, help='Model file path')
    parser.add_argument('--mode', type=str, required=False, help='Model file path')

    args = parser.parse_args()

    # pred = gen_proof(args.output, args.data, args.model, args.mode)
    pred = asyncio.run(gen_proof(args.output, args.data, args.model, args.mode))
    
    print(f"Prediction: {pred}")