import os
import torch

def fuse_cartridges(
    base_weights_path="myriad_moe_hierarchical_weights.pt",
    output_fused_path="myriad_moe_hierarchical_fused.pt",
    slot_mapping=None
):
    """
    将多个独立的微专家卡带（如工房规则卡带），物理熔铸融合进轻量化金字塔底座中！
    slot_mapping: { 插槽编号: "卡带文件路径" }
    """
    # 自动向下兼容检查
    if not os.path.exists(base_weights_path):
        fallback_path = "myriad_moe_25k_weights.pt"
        if os.path.exists(fallback_path):
            print(f"[*] 未找到 {base_weights_path}，自动回退使用底座: {fallback_path}")
            base_weights_path = fallback_path
        else:
            raise FileNotFoundError(f"找不到底座权重: {base_weights_path} 或 {fallback_path}，请先训练！")

    if slot_mapping is None:
        # 🌟 配置你的卡带合体清单 (例如将 16 号插槽烧入工房专属卡带)
        slot_mapping = {
            16: "cartridge_gongfang.pt",
        }

    print("=" * 75)
    print("🧬 启动【Myriad-MoE: 金字塔全息卡带熔铸炉】...")
    print("=" * 75)

    print(f"[*] 正在载入通用底座权重: {base_weights_path} ...")
    fused_weights = torch.load(base_weights_path, map_location="cpu")

    # 动态检测底座包含的神经层数
    num_layers = sum(1 for k in fused_weights.keys() if k.endswith("_lora_A"))
    print(f"[*] 底座检测完毕，共包含 {num_layers} 层分形微专家结构。")

    # 逐个插槽遍历熔铸
    for slot_id, cart_file in slot_mapping.items():
        if not os.path.exists(cart_file):
            print(f"⚠️ 跳过插槽 #{slot_id:02d}：找不到卡带文件 {cart_file}")
            continue

        print(f"\n[*] 正在将卡带《{cart_file}》物理熔铸注入插槽 #{slot_id:02d}...")
        cartridge = torch.load(cart_file, map_location="cpu")
        cart_name = cartridge.get("name", "未知卡带")
        print(f"    - 卡带名称: {cart_name}")

        # 逐层覆盖该插槽切片
        for i in range(num_layers):
            layer_data = cartridge["layers"][i]

            # 1. 覆盖 45 专家的 LoRA A 降维矩阵 [45, 16, 1024]
            fused_weights[f"layer_{i}_lora_A"][slot_id].copy_(layer_data["lora_A"])

            # 2. 覆盖 45 专家的 LoRA B 升维矩阵 [45, 1024, 16]
            fused_weights[f"layer_{i}_lora_B"][slot_id].copy_(layer_data["lora_B"])

            # 3. 精准写入该宗门对应的路由器特征向量 (第 slot_id 行)
            if "router_vec" in layer_data:
                fused_weights[f"layer_{i}_router_cluster"]["weight"][slot_id].copy_(layer_data["router_vec"])

        print(f"    ✔ 插槽 #{slot_id:02d} 熔铸完毕，28层共 1,260 个微专家神经元已完全更替！")

    print(f"\n[*] 正在保存合体固化权重至: {output_fused_path} ...")
    torch.save(fused_weights, output_fused_path)

    fsize_mb = os.path.getsize(output_fused_path) / (1024 * 1024)
    print(f"🎉 熔铸圆满成功！多卡带共存版固化模型已就绪 ➔ {output_fused_path} (文件大小: {fsize_mb:.2f} MB)")
    print("👉 提示：可直接将此文件重命名替换底座，或在 chat 脚本中直接指定加载该文件！")

if __name__ == "__main__":
    fuse_cartridges()