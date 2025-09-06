



class LossData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.forget_len + self.retain_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x, y = self.forget_data[index]
            label = 1
        else:
            x, y = self.retain_data[index - self.forget_len]
            label = 0

        # 🔥 关键：全部返回 tensor 类型
        return x, torch.tensor(y).long(), torch.tensor(label).long()

import torch
import torch.nn.functional as F


def compute_forget_regularization_loss(logits, mode="data_kl", target_distribution="uniform"):
    """
    logits: shape (N, C)
    mode: ["data_kl", "data_entropy", "label_kl", "label_entropy"]
    target_distribution: currently only supports "uniform"
    """
    probs = F.softmax(logits, dim=1)

    if target_distribution == "uniform":
        uniform_target = torch.full_like(probs, 1.0 / probs.size(1))  # (N, C)
        uniform_label = torch.full((probs.size(1),), 1.0 / probs.size(1), device=probs.device)  # (C,)
    else:
        raise NotImplementedError("Only uniform target supported for now.")

    if mode == "data_kl":
        # KL for each sample: mean over batch
        log_probs = F.log_softmax(logits, dim=1)
        kl = F.kl_div(log_probs, uniform_target, reduction="batchmean")
        return kl

    elif mode == "data_entropy":
        log_probs = torch.log(probs + 1e-12)
        entropy = - torch.sum(probs * log_probs, dim=1)  # (N,)
        return entropy.mean()

    elif mode == "label_kl":
        mean_prob = probs.mean(dim=0)  # (C,)
        log_mean_prob = torch.log(mean_prob + 1e-12)
        kl = torch.sum(mean_prob * (log_mean_prob - torch.log(uniform_label)))
        return kl

    elif mode == "label_entropy":
        mean_prob = probs.mean(dim=0)  # (C,)
        log_mean_prob = torch.log(mean_prob + 1e-12)
        entropy = - torch.sum(mean_prob * log_mean_prob)
        return entropy

    else:
        raise ValueError(f"Unknown mode: {mode}")



def training_step_diverse(model, batch, criterion,epoch,coefficient_loss_retain,coefficient_loss_forget_ce,coefficient_loss_forget_data_kl,coefficient_loss_forget_label_kl,warmup_epoch):
    device = next(model.parameters()).device
    images, clabels, labels = batch
    images, clabels, labels = images.to(device), clabels.to(device), labels.to(device)

    out = model(images)

    retain_mask = (labels == 0)
    forget_mask = (labels == 1)

    retain_logits = out[retain_mask]
    retain_clabels = clabels[retain_mask]
    if retain_logits.shape[0] > 0:
        loss_retain = criterion(retain_logits, retain_clabels)
    else:
        loss_retain = torch.tensor(0.0, device=device, requires_grad=True)

    forget_logits = out[forget_mask]
    forget_clabels = clabels[forget_mask]
    if forget_logits.shape[0] > 0:
        loss_forget_ce = criterion(forget_logits, forget_clabels) # cross-entropy
        loss_forget_reg_data_kl = compute_forget_regularization_loss(forget_logits, mode='data_kl')
        loss_forget_reg_label_kl = compute_forget_regularization_loss(forget_logits, mode='label_kl')
    else:
        loss_forget_ce = torch.tensor(0.0, device=device, requires_grad=True)
        loss_forget_reg_data_kl = torch.tensor(0.0, device=device, requires_grad=True)
        loss_forget_reg_label_kl = torch.tensor(0.0, device=device, requires_grad=True)

    if epoch < warmup_epoch: #warmup
        loss = coefficient_loss_retain*loss_retain + coefficient_loss_forget_ce * loss_forget_ce
    else:
        loss = coefficient_loss_retain*loss_retain + coefficient_loss_forget_ce * loss_forget_ce + coefficient_loss_forget_data_kl*loss_forget_reg_data_kl+coefficient_loss_forget_label_kl*loss_forget_reg_label_kl

    return loss, loss_retain, loss_forget_ce,loss_forget_reg_data_kl,loss_forget_reg_label_kl



def FT_NegGradPlus_loss(
    num_epochs,
    model,
    train_retain_dl,
    train_forget_dl,
    test_poison_dl,
    test_clean_dl,
    device,
    learning_rate,
    forget_cover_clean_ratio,
    diverse_loss_type,
    coefficient_loss_retain,
    coefficient_loss_forget_ce,
    coefficient_loss_forget_data_kl,
    coefficient_loss_forget_label_kl,
    save_path='.',  # 默认当前目录保存日志
    model_type='neggrad_plus',
    clip_num=1,
    batch_size=128,
    warmup_epoch=2,
    mask=None,
    **kwargs,
):
    set_all_seeds(42)
    print(f"Unlearning ({model_type}) Result: \n")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    unlearning_data = LossData(forget_data=train_forget_dl.dataset, retain_data=train_retain_dl.dataset)
    generator = torch.Generator()
    generator.manual_seed(42)
    training_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True, pin_memory=True,generator=generator)

    log_csv_path = os.path.join(save_path, f"{model_type}_log_ratio_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}.csv")
    # ✅ 创建并写入表头


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

    batch_log_csv_path = os.path.join(
        save_path,
        f"{model_type}_log_ratio_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}_batch_loss_details.csv"
    )

    # 覆盖写入并写表头
    with open(batch_log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'batch',
            'total_loss', 'retain_ce', 'forget_ce',
            'forget_reg_data_kl', 'forget_reg_label_kl',
            'lr'
        ])

    hist_csv_path = os.path.join(
        save_path,
        f"{model_type}_forget_class_hist_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}.csv"
    )
    hist_before = compute_pred_class_hist(model, train_forget_dl, device, num_classes=10, normalize=True)
    save_hist_to_csv(hist_before, hist_csv_path, row_name="before")

    for epoch in range(num_epochs):
        start = time.time()
        print("epoch", epoch)
        model.train()  # <--- 明确切成训练模式
        for i, batch in enumerate(training_loader):
            loss, loss_retain, loss_forget_ce, loss_forget_reg_data_kl,loss_forget_reg_label_kl = training_step_diverse(model, batch, criterion, epoch, coefficient_loss_retain,
                                  coefficient_loss_forget_ce,coefficient_loss_forget_data_kl,coefficient_loss_forget_label_kl,warmup_epoch)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            if clip_num:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch} | Batch {i}] "
                    f"Total Loss: {loss.item():.4f} | "
                    f"Retain CE: {loss_retain.item():.4f} | "
                    f"Forget CE: {loss_forget_ce.item():.4f} | "
                    f"Forget Reg data kl: {loss_forget_reg_data_kl.item():.4f} | "
                    f"Forget Reg label kl: {loss_forget_reg_label_kl.item():.4f}"
                )

                with open(batch_log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, i,
                        f"{loss.item():.4f}",
                        f"{loss_retain.item():.4f}",
                        f"{loss_forget_ce.item():.4f}",
                        f"{loss_forget_reg_data_kl.item():.4f}",
                        f"{loss_forget_reg_label_kl.item():.4f}",
                        f"{optimizer.param_groups[0]['lr']:.6f}"
                    ])

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
    # 只在最后一个 epoch 统计一次
    hist_after = compute_pred_class_hist(model, train_forget_dl, device, num_classes=10, normalize=True)
    save_hist_to_csv(hist_after, hist_csv_path, row_name="after_with_def")
    plot_target_confidence_distribution(
                model, train_forget_dl, device, num_epochs,
                save_dir=save_path, dataset_name=model_type,
                label_mode='gt', threshold=0.9,y_limit=len(train_forget_dl.dataset),coefficient_loss_forget_ce=coefficient_loss_forget_ce,coefficient_loss_forget_data_kl=coefficient_loss_forget_data_kl,coefficient_loss_forget_label_kl=coefficient_loss_forget_label_kl
            )


def GA_loss(
    num_epochs,
    model,
    train_retain_dl,
    train_forget_dl,
    test_poison_dl,
    test_clean_dl,
    device,
    learning_rate,
    forget_cover_clean_ratio,
    diverse_loss_type,
    coefficient_loss_retain,
    coefficient_loss_forget_ce,
    coefficient_loss_forget_data_kl,
    coefficient_loss_forget_label_kl,
    save_path='.',  # 默认当前目录保存日志
    model_type='neggrad_plus',
    clip_num=1,
    batch_size=128,
    warmup_epoch=2,
    mask=None,
    **kwargs,
):
    set_all_seeds(42)
    print(f"Unlearning ({model_type}) Result: \n")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    unlearning_data = LossData(forget_data=train_forget_dl.dataset, retain_data=[])
    generator = torch.Generator()
    generator.manual_seed(42)
    training_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True, pin_memory=True,generator=generator)

    log_csv_path = os.path.join(save_path, f"{model_type}_log_ratio_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}.csv")
    # ✅ 创建并写入表头


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

    batch_log_csv_path = os.path.join(
        save_path,
        f"{model_type}_log_ratio_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}_batch_loss_details.csv"
    )

    # 覆盖写入并写表头
    with open(batch_log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'batch',
            'total_loss', 'retain_ce', 'forget_ce',
            'forget_reg_data_kl', 'forget_reg_label_kl',
            'lr'
        ])

    hist_csv_path = os.path.join(
        save_path,
        f"{model_type}_forget_class_hist_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}.csv"
    )
    hist_before = compute_pred_class_hist(model, train_forget_dl, device, num_classes=10, normalize=True)
    save_hist_to_csv(hist_before, hist_csv_path, row_name="before")

    for epoch in range(num_epochs):
        start = time.time()
        print("epoch", epoch)
        model.train()  # <--- 明确切成训练模式
        for i, batch in enumerate(training_loader):
            loss, loss_retain, loss_forget_ce, loss_forget_reg_data_kl,loss_forget_reg_label_kl = training_step_diverse(model, batch, criterion, epoch, coefficient_loss_retain,
                                  coefficient_loss_forget_ce,coefficient_loss_forget_data_kl,coefficient_loss_forget_label_kl,warmup_epoch)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            if clip_num:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch} | Batch {i}] "
                    f"Total Loss: {loss.item():.4f} | "
                    f"Retain CE: {loss_retain.item():.4f} | "
                    f"Forget CE: {loss_forget_ce.item():.4f} | "
                    f"Forget Reg data kl: {loss_forget_reg_data_kl.item():.4f} | "
                    f"Forget Reg label kl: {loss_forget_reg_label_kl.item():.4f}"
                )

                with open(batch_log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, i,
                        f"{loss.item():.4f}",
                        f"{loss_retain.item():.4f}",
                        f"{loss_forget_ce.item():.4f}",
                        f"{loss_forget_reg_data_kl.item():.4f}",
                        f"{loss_forget_reg_label_kl.item():.4f}",
                        f"{optimizer.param_groups[0]['lr']:.6f}"
                    ])

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
    # 只在最后一个 epoch 统计一次
    hist_after = compute_pred_class_hist(model, train_forget_dl, device, num_classes=10, normalize=True)
    save_hist_to_csv(hist_after, hist_csv_path, row_name="after_with_def")
    plot_target_confidence_distribution(
                model, train_forget_dl, device, num_epochs,
                save_dir=save_path, dataset_name=model_type,
                label_mode='gt', threshold=0.9,y_limit=len(train_forget_dl.dataset),coefficient_loss_forget_ce=coefficient_loss_forget_ce,coefficient_loss_forget_data_kl=coefficient_loss_forget_data_kl,coefficient_loss_forget_label_kl=coefficient_loss_forget_label_kl
            )

def RL_Salun_loss(
    num_epochs,
    model,
    train_retain_dl,
    train_forget_dl,
    test_poison_dl,
    test_clean_dl,
    device,
    learning_rate,
    forget_cover_clean_ratio,
    diverse_loss_type,
    coefficient_loss_retain,
    coefficient_loss_forget_ce,
    coefficient_loss_forget_data_kl,
    coefficient_loss_forget_label_kl,
    save_path='.',  # 默认当前目录保存日志
    model_type='neggrad_plus',
    clip_num=1,
    batch_size=128,
    warmup_epoch=2,
    mask=None,
    **kwargs,
):
    set_all_seeds(42)
    print(f"Unlearning ({model_type}) Result: \n")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    num_classes = 10
    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    for sample in train_forget_dl.dataset:
        x, clabel = sample[0], sample[1].item()
        rnd = random.choice(unlearninglabels)
        while rnd == clabel:
            rnd = random.choice(unlearninglabels)
        # unlearning_trainset.append((x, rnd))
        unlearning_trainset.append((x, torch.tensor(rnd).long()))

    for sample in train_retain_dl.dataset:
        x, y = sample[0], sample[1].item()
        unlearning_trainset.append((x, torch.tensor(y, dtype=torch.long)))

    g = torch.Generator()
    g.manual_seed(42)
    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, batch_size=batch_size, pin_memory=True, shuffle=True, generator=g
    )
    unlearning_data = LossData(forget_data=train_forget_dl.dataset, retain_data=unlearning_train_set_dl.dataset)

    generator = torch.Generator()
    generator.manual_seed(42)
    training_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True, pin_memory=True,generator=generator)

    log_csv_path = os.path.join(save_path, f"{model_type}_log_ratio_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}.csv")
    # ✅ 创建并写入表头


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

    batch_log_csv_path = os.path.join(
        save_path,
        f"{model_type}_log_ratio_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}_batch_loss_details.csv"
    )

    # 覆盖写入并写表头
    with open(batch_log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'batch',
            'total_loss', 'retain_ce', 'forget_ce',
            'forget_reg_data_kl', 'forget_reg_label_kl',
            'lr'
        ])

    hist_csv_path = os.path.join(
        save_path,
        f"{model_type}_forget_class_hist_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}.csv"
    )
    hist_before = compute_pred_class_hist(model, train_forget_dl, device, num_classes=10, normalize=True)
    save_hist_to_csv(hist_before, hist_csv_path, row_name="before")

    for epoch in range(num_epochs):
        start = time.time()
        print("epoch", epoch)
        model.train()  # <--- 明确切成训练模式
        for i, batch in enumerate(training_loader):
            loss, loss_retain, loss_forget_ce, loss_forget_reg_data_kl,loss_forget_reg_label_kl = training_step_diverse(model, batch, criterion, epoch, coefficient_loss_retain,
                                  coefficient_loss_forget_ce,coefficient_loss_forget_data_kl,coefficient_loss_forget_label_kl,warmup_epoch)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            if clip_num:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch} | Batch {i}] "
                    f"Total Loss: {loss.item():.4f} | "
                    f"Retain CE: {loss_retain.item():.4f} | "
                    f"Forget CE: {loss_forget_ce.item():.4f} | "
                    f"Forget Reg data kl: {loss_forget_reg_data_kl.item():.4f} | "
                    f"Forget Reg label kl: {loss_forget_reg_label_kl.item():.4f}"
                )

                with open(batch_log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, i,
                        f"{loss.item():.4f}",
                        f"{loss_retain.item():.4f}",
                        f"{loss_forget_ce.item():.4f}",
                        f"{loss_forget_reg_data_kl.item():.4f}",
                        f"{loss_forget_reg_label_kl.item():.4f}",
                        f"{optimizer.param_groups[0]['lr']:.6f}"
                    ])

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
    # 只在最后一个 epoch 统计一次
    hist_after = compute_pred_class_hist(model, train_forget_dl, device, num_classes=10, normalize=True)
    save_hist_to_csv(hist_after, hist_csv_path, row_name="after_with_def")
    plot_target_confidence_distribution(
                model, train_forget_dl, device, num_epochs,
                save_dir=save_path, dataset_name=model_type,
                label_mode='gt', threshold=0.9,y_limit=len(train_forget_dl.dataset),coefficient_loss_forget_ce=coefficient_loss_forget_ce,coefficient_loss_forget_data_kl=coefficient_loss_forget_data_kl,coefficient_loss_forget_label_kl=coefficient_loss_forget_label_kl
            )