# ScreenSpot Pro 评估设置
我们从 ScreenSpot Pro (ss pro) 官方[仓库](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/tree/main)中改编了评估代码。此文件夹包含在此基准测试上的推理脚本/结果。我们正在进行法律审查过程以发布 omniparser v2。一旦完成，我们将更新文件以便它可以加载 v2 模型。
1. eval/ss_pro_gpt4o_omniv2.py：包含我们使用的提示，它可以替代原始 ss pro 仓库中的这个[文件](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/blob/main/models/gpt4x.py)。
2. eval/logs_sspro_omniv2.json：包含使用 GPT4o+OmniParserv2 对 ss pro 进行推理的结果。 
