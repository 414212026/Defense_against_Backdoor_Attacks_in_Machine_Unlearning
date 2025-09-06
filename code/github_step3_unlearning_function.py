
#Thanks Yingdan https://arxiv.org/pdf/2505.10859 to share these codes


def training_step(model, batch, criterion, device):
    images, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)  # Generate predictions
    loss = criterion(out, clabels)  # Calculate loss
    _, pred = torch.max(out, 1)
    num_correct = (pred == clabels).sum()
    acc = num_correct.item() / len(clabels)
    return loss, acc, num_correct


def fit_one_cycle(
        epochs, model, train_retain_dl, train_forget_dl,
        test_poison_dl, test_clean_dl, device,
        lr, milestones, forget_cover_clean_ratio,
        mask=None, save_path='.', model_type='Missing', clip_num=None, probe_epochs=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)

    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        if milestones else
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    )

    log_csv_path = os.path.join(save_path, f"{model_type}_log_ratio_{forget_cover_clean_ratio}.csv")
    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_forget_acc',
            'test_poison_acc', 'test_clean_acc',
            'lr', 'epoch_time'
        ])

    base_log_path = os.path.join(save_path, "base_log_epoch-1.csv")
    if base_log_path and os.path.exists(base_log_path):
        with open(base_log_path, mode='r') as base_f:
            lines = list(csv.reader(base_f))
            if len(lines) >= 2:
                epoch_minus1_row = lines[1]
                with open(log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(epoch_minus1_row)
                print("[Info] Inserted epoch=-1 row from base_log.")
            else:
                print("[Warning] base_log_path does not contain epoch -1 row.")

    for epoch in range(epochs):
        start = time.time()
        print("epoch", epoch)
        model.train()  # <--- 明确切成训练模式
        pbar = tqdm(train_retain_dl, total=len(train_retain_dl))

        for batch in pbar:
            loss, acc, correct = training_step(model, batch, criterion, device)
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            if clip_num:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        model.eval()  # <--- 明确切成评估模式
        # Accuracy 评估

        train_forget_acc = evaluate_acc(model, train_forget_dl, device)[1]
        test_poison_acc = evaluate_acc(model, test_poison_dl, device)[1]
        test_clean_acc = evaluate_acc(model, test_clean_dl, device)[1]

        # 打印结果
        print(f"After epoch {epoch}:\n"

              f"Train Forget Acc:  {train_forget_acc:.4f} \n"
              f"Test Poison Acc:   {test_poison_acc:.4f} \n"
              f"Test Clean Acc:    {test_clean_acc:.4f} \n"

              )

            # 写入日志
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_forget_acc,
                test_poison_acc, test_clean_acc,
                optimizer.param_groups[0]['lr'],
                round(time.time() - start, 2),
            ])

        # >>> 新增：如果设置了 probe_epochs，就在达到后 break
        if probe_epochs is not None and (epoch + 1) >= probe_epochs:
            print(f"[Probe Mode] Reached probe_epochs={probe_epochs}, stopping early.")
            break
        # <<< 新增结束




def finetune(
        epochs,
        model,
        train_retain_dl,
        train_forget_dl,
        test_poison_dl,
        test_clean_dl,

        device,
        lr,
        milestones,
        forget_cover_clean_ratio,
        mask=None,
        save_path='.',
        model_type='FT',
        clip_num=None,
        probe_epochs=None,

        **kwargs,
):
    # print(f"Unlearning (Finetune) Result: \n")
    fit_one_cycle(
        epochs=epochs, model=model, train_retain_dl=train_retain_dl, train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl, test_clean_dl=test_clean_dl, device=device,
        lr=lr, milestones=milestones, forget_cover_clean_ratio=forget_cover_clean_ratio, mask=mask,
        save_path=save_path, model_type=model_type, clip_num=clip_num, probe_epochs=probe_epochs
    )


def RL(
        epochs,
        model,
        train_retain_dl,
        train_forget_dl,
        test_poison_dl,
        test_clean_dl,
        device,
        lr,
        milestones,
        forget_cover_clean_ratio,
        mask=None,
        save_path='.',
        model_type='FT',
        clip_num=None,
        num_classes=10,
        batch_size=128,
        probe_epochs=None,
        **kwargs,
):
    import random
    from torch.utils.data import DataLoader

    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    for sample in train_forget_dl.dataset:
        x, clabel = sample[0], sample[1].item()
        rnd = random.choice(unlearninglabels)
        while rnd == clabel:
            rnd = random.choice(unlearninglabels)
        unlearning_trainset.append((x, rnd))

    for sample in train_retain_dl.dataset:
        x, y = sample[0], sample[1].item()
        unlearning_trainset.append((x, y))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    train_retain_dl = unlearning_train_set_dl
    # print(f"Unlearning (RL) Result: \n")
    fit_one_cycle(
        epochs=epochs, model=model, train_retain_dl=train_retain_dl, train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl, test_clean_dl=test_clean_dl, device=device, lr=lr, milestones=milestones,
        forget_cover_clean_ratio=forget_cover_clean_ratio, mask=mask, save_path=save_path, model_type=model_type,
        clip_num=clip_num, probe_epochs=probe_epochs

    )


def GA(
        epochs, model, train_retain_dl, train_forget_dl,
        test_poison_dl, test_clean_dl, device,
        lr, milestones, forget_cover_clean_ratio,
        mask=None, save_path='.', model_type='Missing', clip_num=None, probe_epochs=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)


    log_csv_path = os.path.join(save_path, f"{model_type}_log_ratio_{forget_cover_clean_ratio}.csv")
    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_forget_acc',
            'test_poison_acc', 'test_clean_acc',
            'lr', 'epoch_time'
        ])

    base_log_path = os.path.join(save_path, "base_log_epoch-1.csv")
    if base_log_path and os.path.exists(base_log_path):
        with open(base_log_path, mode='r') as base_f:
            lines = list(csv.reader(base_f))
            if len(lines) >= 2:
                epoch_minus1_row = lines[1]
                with open(log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(epoch_minus1_row)
                print("[Info] Inserted epoch=-1 row from base_log.")
            else:
                print("[Warning] base_log_path does not contain epoch -1 row.")

    for epoch in range(epochs):
        start = time.time()
        print("epoch", epoch)
        model.train()  # <--- 明确切成训练模式

        for i, (image, target) in enumerate(train_forget_dl):
            image = image.to(device)
            target = target.to(device)

            output_clean = model(image)
            loss = - criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()
            if clip_num:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)
            optimizer.step()
        model.eval()  # <--- 明确切成评估模式
        # Accuracy 评估

        train_forget_acc = evaluate_acc(model, train_forget_dl, device)[1]
        test_poison_acc = evaluate_acc(model, test_poison_dl, device)[1]
        test_clean_acc = evaluate_acc(model, test_clean_dl, device)[1]

        # 打印结果
        print(f"After epoch {epoch}:\n"

              f"Train Forget Acc:  {train_forget_acc:.4f} \n"
              f"Test Poison Acc:   {test_poison_acc:.4f} \n"
              f"Test Clean Acc:    {test_clean_acc:.4f} \n"

              )

        # 写入日志
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_forget_acc,
                test_poison_acc, test_clean_acc,
                optimizer.param_groups[0]['lr'],
                round(time.time() - start, 2),
            ])

        # >>> 新增：如果设置了 probe_epochs，就在达到后 break
        if probe_epochs is not None and (epoch + 1) >= probe_epochs:
            print(f"[Probe Mode] Reached probe_epochs={probe_epochs}, stopping early.")
            break
        # <<< 新增结束




class LossData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x, y = self.forget_data[index]
            label = 1
            return x, y, label
        else:
            x, y = self.retain_data[index - self.forget_len]
            label = 0
            return x, y, label


### ga_plus
def training_step_ga_plus(model, batch, criterion):
    device = next(model.parameters()).device
    images, clabels, labels = batch
    images, clabels, labels = images.to(device), clabels.to(device), labels.to(device)

    out = model(images)

    retain_mask = (labels == 0)
    forget_mask = (labels == 1)

    retain_logits = out[retain_mask]
    retain_clabels = clabels[retain_mask]
    loss_retain = criterion(retain_logits, retain_clabels)

    forget_logits = out[forget_mask]
    forget_clabels = clabels[forget_mask]
    loss_forget = criterion(forget_logits, forget_clabels)

    loss = loss_retain - 0.15 * loss_forget  # Calculate loss
    return loss, loss_retain, loss_forget


def neggrad_plus(
        epochs, model, train_retain_dl, train_forget_dl,
        test_poison_dl, test_clean_dl, device,
        lr, milestones, forget_cover_clean_ratio,
        mask=None, save_path='.', model_type='Missing', clip_num=None, probe_epochs=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    unlearning_data = LossData(forget_data=train_forget_dl.dataset, retain_data=train_retain_dl.dataset)
    training_loader = DataLoader(unlearning_data, batch_size=128, shuffle=True, pin_memory=True)


    log_csv_path = os.path.join(save_path, f"{model_type}_log_ratio_{forget_cover_clean_ratio}.csv")
    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_forget_acc',
            'test_poison_acc', 'test_clean_acc',
            'lr', 'epoch_time'
        ])

    base_log_path = os.path.join(save_path, "base_log_epoch-1.csv")
    if base_log_path and os.path.exists(base_log_path):
        with open(base_log_path, mode='r') as base_f:
            lines = list(csv.reader(base_f))
            if len(lines) >= 2:
                epoch_minus1_row = lines[1]
                with open(log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(epoch_minus1_row)
                print("[Info] Inserted epoch=-1 row from base_log.")
            else:
                print("[Warning] base_log_path does not contain epoch -1 row.")

    for epoch in range(epochs):
        start = time.time()
        print("epoch", epoch)
        model.train()  # <--- 明确切成训练模式
        for i, batch in enumerate(training_loader):
            loss, _, _ = training_step_ga_plus(model, batch, criterion)
            optimizer.zero_grad()
            loss.backward()
            if clip_num:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)
            optimizer.step()
        scheduler.step()
        model.eval()  # <--- 明确切成评估模式
        # Accuracy 评估

        train_forget_acc = evaluate_acc(model, train_forget_dl, device)[1]
        test_poison_acc = evaluate_acc(model, test_poison_dl, device)[1]
        test_clean_acc = evaluate_acc(model, test_clean_dl, device)[1]

        # 打印结果
        print(f"After epoch {epoch}:\n"

              f"Train Forget Acc:  {train_forget_acc:.4f} \n"
              f"Test Poison Acc:   {test_poison_acc:.4f} \n"
              f"Test Clean Acc:    {test_clean_acc:.4f} \n"

              )

        # 写入日志
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_forget_acc,
                test_poison_acc, test_clean_acc,
                optimizer.param_groups[0]['lr'],
                round(time.time() - start, 2),
            ])

        # >>> 新增：如果设置了 probe_epochs，就在达到后 break
        if probe_epochs is not None and (epoch + 1) >= probe_epochs:
            print(f"[Probe Mode] Reached probe_epochs={probe_epochs}, stopping early.")
            break
        # <<< 新增结束




def save_gradient_ratio(data_loaders, model, criterion, args, clip_num, forget_cover_clean_ratio):
    """
    Generate gradient-based masks and save them.
    data_loaders: dict with keys 'forget' and optional 'retain'
    model: the torch model
    criterion: loss function (e.g., CrossEntropyLoss)
    args: Namespace or dict with attributes:
        unlearn_lr, momentum, weight_decay, save_dir
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    gradients = {}
    forget_loader = data_loaders["forget"]
    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = 0

    for i, batch in enumerate(forget_loader):
        image = batch[0]
        target = batch[1]
        image = image.cuda()
        target = target.cuda()

        output_clean = model(image)
        loss = -criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        if clip_num:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)
        optimizer.step()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs(gradients[name])

    # threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    threshold_list = [0.5]

    for i in threshold_list:
        sorted_dict_positions = {}
        hard_dict = {}

        all_elements = torch.cat([tensor.flatten() for tensor in gradients.values()])
        threshold_index = int(len(all_elements) * i)

        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            tensor_ranks = ranks[start_index: start_index + num_elements]
            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        all_gradients = torch.cat([gradient.flatten() for gradient in gradients.values()])
        sigmoid_gradients = torch.abs(2 * (torch.sigmoid(all_gradients) - 0.5))
        tanh_gradients = torch.abs(torch.tanh(all_gradients))

        sigmoid_soft_dict = {}
        tanh_soft_dict = {}

        start_idx = 0
        for net_name, gradient in gradients.items():
            num_params = gradient.numel()
            end_idx = start_idx + num_params

            sigmoid_gradient = sigmoid_gradients[start_idx:end_idx].reshape(gradient.shape)
            sigmoid_soft_dict[net_name] = sigmoid_gradient

            tanh_gradient = tanh_gradients[start_idx:end_idx].reshape(gradient.shape)
            tanh_soft_dict[net_name] = tanh_gradient

            start_idx = end_idx

        # Save masks
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(sigmoid_soft_dict, os.path.join(args.save_dir, f"sigmoid_soft_mask_{forget_cover_clean_ratio}.pt"))
        torch.save(tanh_soft_dict, os.path.join(args.save_dir, f"tanh_soft_mask_{forget_cover_clean_ratio}.pt"))
        torch.save(hard_dict, os.path.join(args.save_dir, f"hard_mask_{forget_cover_clean_ratio}.pt"))


def salun(
        epochs,
        model,
        train_retain_dl,
        train_forget_dl,
        test_poison_dl,
        test_clean_dl,
        device,
        lr,
        milestones,
        forget_cover_clean_ratio,
        mask=None,
        save_path='.',
        model_type='FT',
        clip_num=None,
        num_classes=10,
        batch_size=128,
        probe_epochs=None,
        **kwargs,
):
    import random
    from torch.utils.data import DataLoader

    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    for sample in train_forget_dl.dataset:
        x, clabel = sample[0], sample[1].item()
        rnd = random.choice(unlearninglabels)
        while rnd == clabel:
            rnd = random.choice(unlearninglabels)
        unlearning_trainset.append((x, rnd))

    for sample in train_retain_dl.dataset:
        x, y = sample[0], sample[1].item()
        unlearning_trainset.append((x, y))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    train_retain_dl = unlearning_train_set_dl
    # print(f"Unlearning (salun) Result: \n")
    fit_one_cycle(
        epochs=epochs, model=model, train_retain_dl=train_retain_dl, train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl, test_clean_dl=test_clean_dl, device=device, lr=lr, milestones=milestones,
        forget_cover_clean_ratio=forget_cover_clean_ratio, mask=mask, save_path=save_path, model_type=model_type,
        clip_num=clip_num, probe_epochs=probe_epochs

    )