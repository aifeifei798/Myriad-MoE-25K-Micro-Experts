import os
import torch

def fuse_cartridges(
    base_weights_path="myriad_moe_25k_weights.pt",
    output_fused_path="myriad_moe_25k_ultimate_fused.pt",
    slot_mapping=None
):
    """
    将多个独立的微专家卡带，批量熔铸融合进底座大权重中！
    slot_mapping: { 插槽编号: "卡带文件路径" }
    """
    assert os.path.exists(base_weights_path), f"找不到底座权重: {base_weights_path}"
    
    if slot_mapping is None:
        # 🌟 配置你的卡带合体清单：
        slot_mapping = {
            16: "cartridge_gongfang.pt",     # 16 号插槽烧入：工房1规则
        }

    print("=" * 70)
    print("🧬 启动【Myriad-MoE 模块化卡带全息融合工坊 (修复版)】...")
    print("=" * 70)

    print(f"[*] 正在载入通用万象底座: {base_weights_path} (1.25 GB)...")
    fused_weights = torch.load(base_weights_path, map_location="cpu")

    # 逐个插槽遍历融合
    for slot_id, cart_file in slot_mapping.items():
        if not os.path.exists(cart_file):
            print(f"⚠️ 跳过插槽 #{slot_id:02d}：找不到卡带文件 {cart_file}")
            continue

        print(f"[*] 正在将卡带《{cart_file}》熔铸注入插槽 #{slot_id:02d}...")
        cartridge = torch.load(cart_file, map_location="cpu")

        # 遍历 28 层，精准覆盖该插槽的切片
        for i in range(28):
            layer_data = cartridge["layers"][i]
            
            # 1. 替换 LoRA A 矩阵 [45, 16, 1024]
            fused_weights[f"layer_{i}_lora_A"][slot_id] = layer_data["lora_A"]
            
            # 2. 替换 LoRA B 矩阵 [45, 1024, 16]
            fused_weights[f"layer_{i}_lora_B"][slot_id] = layer_data["lora_B"]
            
            # 3. 🌟 修复核心：必须精准写入 "weight" 张量的第 slot_id 行！
            if "router_vec" in layer_data:
                fused_weights[f"layer_{i}_router_cluster"]["weight"][slot_id] = layer_data["router_vec"]

        print(f"    ✔ 插槽 #{slot_id:02d} 融合完毕！")

    print(f"\n[*] 正在保存全量合体大模型至: {output_fused_path} ...")
    torch.save(fused_weights, output_fused_path)
    print(f"🎉 融合圆满成功！多卡带共存版模型已就绪 ➔ {output_fused_path}！")

if __name__ == "__main__":
    fuse_cartridges()