import asyncio
import os
import shutil
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
    run_args.num_inner_cols = 2
    run_args.variables = [("batch_size", 1)]
    run_args.input_scale = 12
    run_args.param_scale = 12
    run_args.scale_rebase_multiplier = 1
    lsm = 8
    max_log_rows = 18

    print("Generating settings")
    try:
        # res = ezkl.gen_settings(model_path, settings_path, py_run_args=run_args)
        command = f'ezkl gen-settings --model {model_path} --settings-path {settings_path} \
            --input-scale {run_args.input_scale} --param-scale {run_args.param_scale} --scale-rebase-multiplier {run_args.scale_rebase_multiplier} \
            --num-inner-cols {run_args.num_inner_cols} --input-visibility {run_args.input_visibility} --output-visibility {run_args.output_visibility} --param-visibility {run_args.param_visibility}'
        os.system(command)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    # assert res == True

    print("Calibrating settings")
    try:
        # res = await ezkl.calibrate_settings(data_path, model_path, settings_path, mode, scales=[run_args.input_scale], lookup_safety_margin=lsm)
        command = f'ezkl calibrate-settings --data "{data_path}" --model "{model_path}" --settings-path "{settings_path}" --target {mode} --scales {run_args.input_scale} \
                --scale-rebase-multiplier {run_args.scale_rebase_multiplier} --lookup-safety-margin {lsm} --max-logrows {max_log_rows}'
        os.system(command)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    # assert res == True

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
    # remove cache file to have fair comparison.
    cache_dir = os.path.expanduser("~/.ezkl/cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

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