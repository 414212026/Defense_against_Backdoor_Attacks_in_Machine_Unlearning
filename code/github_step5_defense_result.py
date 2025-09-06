


data = torch.load(os.path.join(save_path, "final_all_datasets.pt"))

# 基础数据集（如果你之后需要重新生成 dataloader）
bd_train = data["bd_train"]
clean_test = data["clean_test"]
bd_test = data["bd_test"]

# 构建 DataLoader
# For reproduction, I use random.shuffle(indices) with seed=42 to create final_all_datasets.pt for train data, so below shuffle=False
train_retain_dl = DataLoader(
    Subset(data["retain_dataset"], data["retain_indices"]),
    batch_size=128, shuffle=False, num_workers=0, pin_memory=True
)

# For reproduction, I use random.shuffle(indices) with seed=42 to create final_all_datasets.pt for train data, so below shuffle=False
train_forget_dl = DataLoader(
    Subset(data["forget_dataset"], data["forget_indices"]),
    batch_size=128, shuffle=False, num_workers=0, pin_memory=True
)

# For reproduction, I use random.shuffle(indices) with seed 42 to create final_all_datasets.pt for train data, so below shuffle=False
dl_poison_train = DataLoader(
    data["poison_dataset"],
    batch_size=128, shuffle=False, num_workers=0, pin_memory=True
)

test_poison_dl = DataLoader(
    data["test_poison_dataset"],
    batch_size=128, shuffle=False, num_workers=0, pin_memory=True
)

test_clean_dl = DataLoader(
    data["test_clean_dataset"],
    batch_size=128, shuffle=False, num_workers=0, pin_memory=True
)



def write_initial_log_entry(model, device, log_csv_path,
                            train_forget_dl, test_poison_dl, test_clean_dl):
    # 在追加模式下写入 epoch=-1
    with open(log_csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)

        train_forget_acc = evaluate_acc(model, train_forget_dl, device)[1]
        test_poison_acc = evaluate_acc(model, test_poison_dl, device)[1]
        test_clean_acc = evaluate_acc(model, test_clean_dl, device)[1]

        print(f"Before unlearning: \n"

              f"Train Forget Acc:  {train_forget_acc:.4f} \n"
              f"Test Poison Acc:   {test_poison_acc:.4f} \n"
              f"Test Clean Acc:    {test_clean_acc:.4f} \n"

              )
        writer.writerow([
            -1, train_forget_acc,
            test_poison_acc, test_clean_acc,
            0.0, 0.0
        ])


 # saved result before unlearning
set_all_seeds(42)
model = generate_cls_model(model_name="preactresnet18", num_classes=10, image_size=32)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

# log path
log_csv_path = os.path.join(save_path, f"base_log_epoch-1.csv")
# 写入表头 + epoch -1 行
with open(log_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'epoch', 'train_forget_acc',
            'test_poison_acc', 'test_clean_acc',
            'lr', 'epoch_time'
    ])

write_initial_log_entry(model, device, log_csv_path,
                             train_forget_dl, test_poison_dl, test_clean_dl)

# ====================================



# ================== 共享设置 ==================
shared_lr_candidates = [0.01,0.001,0.005, 0.02]
probe_epochs = 2
F_drop_thresh = 0.20
ASR_thresh = 0.50
CLEAN_MAX_DROP = 0.10
num_classes = 10

acc_before_forget = evaluate_forget_acc(model, train_forget_dl, device)
clean_acc_before = evaluate_clean_acc(model, test_clean_dl, device)

log_and_print(f"train_retain_dl: {len(train_retain_dl.dataset)}")
log_and_print(f"train_forget_dl: {len(train_forget_dl.dataset)}")
log_and_print(f"dl_poison_train: {len(dl_poison_train.dataset)}")
log_and_print(f"test_poison_dl: {len(test_poison_dl.dataset)}")
log_and_print(f"test_clean_dl: {len(test_clean_dl.dataset)}")
# ================== 工具：评估 + 探针函数（每种方法一份） ==================
def _eval_and_log(tag, lr_try):
    acc_after_forget = evaluate_forget_acc(model, train_forget_dl, device)
    clean_acc_after = evaluate_clean_acc(model, test_clean_dl, device)
    F = compute_F_score(acc_before_forget, acc_after_forget, num_classes)
    ASR = evaluate_asr(model, test_poison_dl, device)
    clean_drop_pp = clean_acc_before - clean_acc_after
    log_and_print(f"[Phase-A Probe][{tag}] lr={lr_try:.6f} | F={F:.2f} | ASR={ASR:.2f} | CleanΔ={clean_drop_pp:.2f}")
    ok = (F >= F_drop_thresh) and (ASR >= ASR_thresh) and (clean_drop_pp <= CLEAN_MAX_DROP)
    return ok


# ---------- FT ----------
def try_lr_list_FT(lr_list):
    for lr_try in lr_list:
        set_all_seeds(BASE_SEED)
        model.load_state_dict(init_state)
        finetune(
            epochs=10,  # 探针期内部仍用 probe_epochs 控制
            model=model,
            train_retain_dl=train_retain_dl,
            train_forget_dl=train_forget_dl,
            test_poison_dl=test_poison_dl,
            test_clean_dl=test_clean_dl,

            device=device,
            lr=lr_try,
            milestones=None,
            forget_cover_clean_ratio=forget_cover_clean_ratio,
            mask=None,
            save_path=save_path,
            model_type=f'FT-probe-lr{lr_try}',
            clip_num=2,
            probe_epochs=probe_epochs,
        )
        if _eval_and_log("FT", lr_try):
            log_and_print(f"[Phase-A SUCCESS][FT] lr* = {lr_try:.6f}")
            return lr_try
    return None


# ---------- RL ----------
def try_lr_list_RL(lr_list):
    for lr_try in lr_list:
        set_all_seeds(BASE_SEED)
        model.load_state_dict(init_state)
        RL(
            epochs=10,  # 探针期
            model=model,
            train_retain_dl=train_retain_dl,
            train_forget_dl=train_forget_dl,
            test_poison_dl=test_poison_dl,
            test_clean_dl=test_clean_dl,

            device=device,
            lr=lr_try,
            milestones=None,
            forget_cover_clean_ratio=forget_cover_clean_ratio,
            mask=None,
            save_path=save_path,
            model_type=f'RL-probe-lr{lr_try}',
            clip_num=2,
            probe_epochs=probe_epochs,
            num_classes=10,
            batch_size=128
        )
        if _eval_and_log("RL", lr_try):
            log_and_print(f"[Phase-A SUCCESS][RL] lr* = {lr_try:.6f}")
            return lr_try
    return None


# ---------- GA ----------
def try_lr_list_GA(lr_list):
    for lr_try in lr_list:
        set_all_seeds(BASE_SEED)
        model.load_state_dict(init_state)
        GA(
            epochs=5,  # 探针期（GA全程就是5）
            model=model,
            train_retain_dl=train_retain_dl,
            train_forget_dl=train_forget_dl,
            test_poison_dl=test_poison_dl,
            test_clean_dl=test_clean_dl,

            device=device,
            lr=lr_try,
            milestones=None,
            forget_cover_clean_ratio=forget_cover_clean_ratio,
            mask=None,
            save_path=save_path,
            model_type=f'GA-probe-lr{lr_try}',
            clip_num=2,
            probe_epochs=probe_epochs,
        )
        if _eval_and_log("GA", lr_try):
            log_and_print(f"[Phase-A SUCCESS][GA] lr* = {lr_try:.6f}")
            return lr_try
    return None


# ---------- NegGradPlus ----------
def try_lr_list_NegGradPlus(lr_list):
    for lr_try in lr_list:
        set_all_seeds(BASE_SEED)
        model.load_state_dict(init_state)
        neggrad_plus(
            epochs=5,  # 探针期（NegGradPlus全程就是5）
            model=model,
            train_retain_dl=train_retain_dl,
            train_forget_dl=train_forget_dl,
            test_poison_dl=test_poison_dl,
            test_clean_dl=test_clean_dl,

            device=device,
            lr=lr_try,
            milestones=None,
            forget_cover_clean_ratio=forget_cover_clean_ratio,
            mask=None,
            save_path=save_path,
            model_type=f'NegGradPlus-probe-lr{lr_try}',
            clip_num=2,
            probe_epochs=probe_epochs,
        )
        if _eval_and_log("NegGradPlus", lr_try):
            log_and_print(f"[Phase-A SUCCESS][NegGradPlus] lr* = {lr_try:.6f}")
            return lr_try
    return None


# ---------- SaLun ----------

def try_lr_list_SaLun(lr_list):
    for lr_try in lr_list:
        set_all_seeds(BASE_SEED)
        model.load_state_dict(init_state)
        salun(
            epochs=10,  # 探针期
            model=model,
            train_retain_dl=train_retain_dl,
            train_forget_dl=train_forget_dl,
            test_poison_dl=test_poison_dl,
            test_clean_dl=test_clean_dl,

            device=device,
            lr=lr_try,
            milestones=None,
            forget_cover_clean_ratio=forget_cover_clean_ratio,
            mask=mask_hard_mask,
            save_path=save_path,
            model_type=f'SaLun-probe-lr{lr_try}',
            clip_num=2,
            probe_epochs=probe_epochs,
            num_classes=10,
            batch_size=128
        )
        if _eval_and_log("SaLun", lr_try):
            log_and_print(f"[Phase-A SUCCESS][SaLun] lr* = {lr_try:.6f}")
            return lr_try
    return None


# ================== 四段“各自独立”的主流程 ==================

# --- FT ---

lr_star = try_lr_list_FT(shared_lr_candidates)
if lr_star is None:
    expanded_list = [v * 3.0 for v in shared_lr_candidates]
    log_and_print(f"[Phase-A][FT] No LR succeeded in {shared_lr_candidates}. Expanding ×3 -> {expanded_list}")
    lr_star = try_lr_list_FT(expanded_list)

if lr_star is None:
    log_and_print(f"[Phase-A FAIL][FT] No LR met all criteria.")
else:
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    finetune(
        epochs=10,  # 正式期 FT=10
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,

        device=device,
        lr=lr_star,
        milestones=None,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        mask=None,
        save_path=save_path,
        model_type='FT',
        clip_num=2,
        probe_epochs=10,
    )
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    FT_NegGradPlus_loss(
        num_epochs=10,
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,
        device=device,
        learning_rate=lr_star,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        diverse_loss_type='kl',
        coefficient_loss_retain=1,
        coefficient_loss_forget_ce=0,
        coefficient_loss_forget_data_kl=10,
        coefficient_loss_forget_label_kl=10,
        save_path=save_path,  # 默认当前目录保存日志
        model_type='FT',
        clip_num=2,
        batch_size=128, warmup_epoch=1
    )

# --- RL ---
lr_star = try_lr_list_RL(shared_lr_candidates)

if lr_star is None:
    expanded_list = [v * 3.0 for v in shared_lr_candidates]
    log_and_print(f"[Phase-A][RL] No LR succeeded in {shared_lr_candidates}. Expanding ×3 -> {expanded_list}")
    lr_star = try_lr_list_RL(expanded_list)

if lr_star is None:
    log_and_print(f"[Phase-A FAIL][RL] No LR met all criteria.")
else:
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    RL(
        epochs=10,  # 正式期 RL=10
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,

        device=device,
        lr=lr_star,
        milestones=None,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        mask=None,
        save_path=save_path,
        model_type='RL',
        clip_num=2,
        probe_epochs=10,
        num_classes=10,
        batch_size=128
    )
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)

    RL_Salun_loss(
        num_epochs=10,
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,
        device=device,
        learning_rate=lr_star,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        diverse_loss_type='kl',
        coefficient_loss_retain=1,
        coefficient_loss_forget_ce=0,
        coefficient_loss_forget_data_kl=10,
        coefficient_loss_forget_label_kl=10,
        save_path=save_path,  # 默认当前目录保存日志
        model_type='RL',
        clip_num=2,
        batch_size=128, warmup_epoch=1
    )

# --- GA ---
lr_star = try_lr_list_GA(shared_lr_candidates)

if lr_star is None:
    expanded_list = [v * 3.0 for v in shared_lr_candidates]
    log_and_print(f"[Phase-A][GA] No LR succeeded in {shared_lr_candidates}. Expanding ×3 -> {expanded_list}")
    lr_star = try_lr_list_GA(expanded_list)

if lr_star is None:
    log_and_print(f"[Phase-A FAIL][GA] No LR met all criteria.")
else:
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    GA(
        epochs=5,  # 正式期 GA=5
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,

        device=device,
        lr=lr_star,
        milestones=None,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        mask=None,
        save_path=save_path,
        model_type='GA',
        clip_num=2,
        probe_epochs=5,
    )
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    GA_loss(
        num_epochs=5,
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,
        device=device,
        learning_rate=lr_star,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        diverse_loss_type='kl',
        coefficient_loss_retain=0,
        coefficient_loss_forget_ce=-1,
        coefficient_loss_forget_data_kl=10,
        coefficient_loss_forget_label_kl=10,
        save_path=save_path,  # 默认当前目录保存日志
        model_type='GA',
        clip_num=2,
        batch_size=128, warmup_epoch=1
    )

# --- NegGradPlus ---
lr_star = try_lr_list_NegGradPlus(shared_lr_candidates)

if lr_star is None:
    expanded_list = [v * 3.0 for v in shared_lr_candidates]
    log_and_print(f"[Phase-A][NegGradPlus] No LR succeeded in {shared_lr_candidates}. Expanding ×3 -> {expanded_list}")
    lr_star = try_lr_list_NegGradPlus(expanded_list)

if lr_star is None:
    log_and_print(f"[Phase-A FAIL][NegGradPlus] No LR met all criteria.")
else:
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    neggrad_plus(
        epochs=5,  # 正式期 NegGradPlus=5
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,

        device=device,
        lr=lr_star,
        milestones=None,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        mask=None,
        save_path=save_path,
        model_type='NegGradPlus',
        clip_num=2,
        probe_epochs=5,
    )
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)

    FT_NegGradPlus_loss(
        num_epochs=5,
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,
        device=device,
        learning_rate=lr_star,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        diverse_loss_type='kl',
        coefficient_loss_retain=1,
        coefficient_loss_forget_ce=-0.15,
        coefficient_loss_forget_data_kl=10,
        coefficient_loss_forget_label_kl=10,
        save_path=save_path,  # 默认当前目录保存日志
        model_type='NegGradPlus',
        clip_num=2,
        batch_size=128, warmup_epoch=1
    )


# --- SaLun ---

# generate mask one time use GA's input

from types import SimpleNamespace

mask_args = SimpleNamespace(
    unlearn_lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    save_dir=save_path,
)
criterion = nn.CrossEntropyLoss()
data_loaders = {"forget": train_forget_dl}
set_all_seeds(BASE_SEED)
model.load_state_dict(init_state)  # 你的 checkpoint 快照
model.to(device)
model.eval()
save_gradient_ratio(data_loaders, model, criterion, args=mask_args, clip_num=2,
                    forget_cover_clean_ratio=forget_cover_clean_ratio)
mask_path_hard_mask = os.path.join(save_path, f"hard_mask_{forget_cover_clean_ratio}.pt")
mask_hard_mask = torch.load(mask_path_hard_mask)

lr_star = try_lr_list_SaLun(shared_lr_candidates)

if lr_star is None:
    expanded_list = [v * 3.0 for v in shared_lr_candidates]
    log_and_print(f"[Phase-A][SaLun] No LR succeeded in {shared_lr_candidates}. Expanding ×3 -> {expanded_list}")
    lr_star = try_lr_list_SaLun(expanded_list)

if lr_star is None:
    log_and_print(f"[Phase-A FAIL][SaLun] No LR met all criteria.")
else:
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    salun(
                epochs=10,  # 正式期 salun=10
                model=model,
                train_retain_dl=train_retain_dl,
                train_forget_dl=train_forget_dl,
                test_poison_dl=test_poison_dl,
                test_clean_dl=test_clean_dl,

                device=device,
                lr=lr_star,
                milestones=None,
                forget_cover_clean_ratio=forget_cover_clean_ratio,
                mask=mask_hard_mask,
                save_path=save_path,
                model_type='SaLun',
                clip_num=2,
                probe_epochs=10,
                num_classes=10,
                batch_size=128
            )

    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    RL_Salun_loss(
        num_epochs=10,
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,
        device=device,
        learning_rate=lr_star,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        diverse_loss_type='kl',
        coefficient_loss_retain=1,
        coefficient_loss_forget_ce=0,
        coefficient_loss_forget_data_kl=10,
        coefficient_loss_forget_label_kl=10,
        save_path=save_path,  # 默认当前目录保存日志
        model_type='Salun',
        clip_num=2,
        batch_size=128, warmup_epoch=1, mask=mask_hard_mask
    )