import copy

def f_ack(cob,input_data=None, original_pred=None, layer_idx=None ,original_loss=None, tm=None):    
    
    teleported_model = tm
    
    # apply the COB
    teleported_model = teleported_model.teleport(cob, reset_teleportation=False)

    # Reset activation stats and run a forward pass
    activation_stats = {}
    hook_handles = []
    for i, layer in enumerate(teleported_model.network.children()):
            if isinstance(layer, nn.ReLU) or isinstance(layer, ReLUCOB) or isinstance(layer, SigmoidCOB) or isinstance(layer, nn.Sigmoid) or isinstance(layer, GELUCOB) or isinstance(layer, nn.GELU) or isinstance(layer, LeakyReLUCOB) or isinstance(layer, nn.LeakyReLU):
                handle = layer.register_forward_hook(activation_hook(f'relu_{i}',activation_stats=activation_stats))
                hook_handles.append(handle)
    teleported_model.eval()
    with torch.no_grad():
        pred = teleported_model.network(input_data)
    for handle in hook_handles:
        handle.remove()

    # Calculate the range loss
    loss = sum([stats['max'] - stats['min'] for stats in activation_stats.values()])
    loss /= original_loss
    # Calculate the prediction error
    pred_error = torch.abs(original_pred - pred).mean()
    pred_error /= np.abs(original_pred).mean()
    total_loss = loss + args.pred_mul * pred_error

    if random.random() < 0.0005:
         print(f"pred_error: {pred_error} \t range_loss: {loss}")
    
    # Undo the teleportation
    teleported_model.undo_teleportation()

    # activation_stats.clear()
    # del teleported_model, activation_stats, pred, loss, pred_error, cob
    return total_loss.detach().cpu().numpy()


@torch.no_grad()
def cge(func, params_dict, mask_dict, step_size, base=None, mask_ratio=0.5):
    if base == None:
        base = func(params_dict["cob"])
    grads_dict = {}
    for key, param in params_dict.items():
        if 'orig' in key:
            mask_key = key.replace('orig', 'mask')
            mask_flat = mask_dict[mask_key].flatten()
        else:
            mask_flat = torch.ones_like(param).flatten()
        directional_derivative = torch.zeros_like(param)
        directional_derivative_flat = directional_derivative.flatten()
        # set 50 percent of the mask to zero
        mask_flat = mask_flat * torch.bernoulli(torch.full_like(mask_flat, mask_ratio))

        perturbed_params_dict = copy.deepcopy(params_dict)
        p_flat = perturbed_params_dict[key].flatten()
        
        for idx in mask_flat.nonzero().flatten():
            p_flat[idx] += step_size
            directional_derivative_flat[idx] = (func(perturbed_params_dict["cob"]) - base) / step_size
            p_flat[idx] -= step_size
        grads_dict[key] = directional_derivative.to(param.device)
    return grads_dict


if __name__ == "__main__":
    # some lines of codes which are not shown here (not related to the multiprocessing)
    initial_cob_idx = torch.ones(960)
    # add inputs to the function
    ackley = functools.partial(
        f_ack,
        input_data=input_teleported_model,
        original_pred=original_pred,
        layer_idx=layer_idx,
        original_loss = original_loss_idx,\
        tm = LN
    )

    # training to find best_cob
    best_cob = None
    for step in range(args.steps):
        # get the gradient of the cob
        grad_cob = cge(ackley, {"cob": initial_cob_idx}, None, args.zoo_step_size, mask_ratio=args.mask_ratio)
        # update the cob
        initial_cob_idx -= args.cob_lr * grad_cob["cob"]
        # calculate the loss
        loss = ackley(initial_cob_idx)
        # update the best loss
        if loss < best_loss:
            best_loss = loss
            best_cob = initial_cob_idx
            print(f"Step: {step} \t Loss: {loss}")

    print("BEST LOSS:",best_loss)