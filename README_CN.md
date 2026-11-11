# Math-31846 全部科研代码与复现记录

本包对应 2026-11-11 的返修稿，包含原始 12 个脚本、经核查的 12 个修正版，以及独立复算、完整 FFT 交叉检查、长时间检查和当前图 2 的重绘代码。原始版本和修正版均保留，以便逐项追溯。所有科研脚本合计 34 份；另有 1 份新增的文件完整性检查脚本。

## 先使用哪些目录

- `evidence/`：当前论文表格和新增检查对应的独立代码，优先使用。
- `patched_project/`：原项目 12 个脚本的核查修正版；保留原来的数字文件名和导入关系。
- `original_code/`：你上传的 12 个脚本原样保存，含已发现的历史问题，不作为当前复算的默认入口。
- `evidence/final_results/`、`evidence/extended_results/`：已完成计算的全部相应状态文件和数据。
- `archived_results/`、`audit_evidence/`、`patched_quick_results/`：原始结果、审计记录、短时间 quick 预设结果。
- `audit_workflow/`：原版/修正版核查脚本及前一阶段独立长时间工作流。与 evidence 中重复的求解器和长时间脚本保留两个路径，以保证各自工作流可运行。
- `figures/`：当前修订稿使用的两张研究图片。
- `docs/`：GitHub 发布说明、来源记录、返修追溯快照。

这是科研计算代码包，不包含文档排版自动化脚本或第三方运行库。

## 运行

克隆仓库或解压下载文件后，进入包含 `README.md` 和 `verify_package.py` 的仓库根目录。不要把数字命名脚本重命名或分散到不同目录。

1. `python verify_package.py`：核对全部文件校验值、Python 语法和源码数量。
2. `python -m pip install -r requirements.txt`：安装独立复算需要的 NumPy、SciPy、Matplotlib 和 mpmath。原项目和 PyTorch 核查另需安装适配本机的 PyTorch，已测试版本见 `requirements_torch_tested.txt`。
3. 按英文 `README.md` 中逐项命令运行。它明确列出工作目录，Windows/PyCharm 终端也可以照此执行。

本轮没有重新运行所有长时间 PDE 实验。已保存的正式结果原样收入；本次打包仅重新做文件、语法、短程求解器及绘图入口检查。具体结果见 `docs/PACKAGE_VALIDATION.json`。

当前表 2、替代表 3 使用 `offset` 初始化；`critical` 初始化用于对照审稿人的实现。完整网格点数 M 对应偶延拓区间 [0,2L)，与半区间节点数不要混用。

原项目 7.1.py 的 quick 预设与 paper 预设都不能直接替代当前表 3 的基准设置。2.py 的空间部分只有网格配置，不是空间收敛结果。当前表 3 请运行 evidence/reproduce_concern1.py。

图 2 请使用 evidence/regenerate_branch_figure.py；图 1 请运行 evidence/R2_benchmark_precision_check.py --output rerun_outputs/reviewer2。旧绘图脚本保留用于追溯，其旧标签不应替代当前正文图件。

## 与当前返修的关系

旧表 3 的数值及其定量空间收敛含义已撤回。旧值的具体生成依据仍未查明，本包没有将其归因于未经证实的程序或抄录错误。

表 2、图 2 是有限时间一致性检查。初始化已经使用理论领先阶振幅。延长时间后，驻定残差下降，但系数偏差由约 0.0813% 增至约 0.1677%，两项结果均保留。不能据此声称独立验证了渐近系数。

返修定位快照对应 35 页论文。其证据相对路径保留可查；历史审计日志中的状态属于当时的数值审计，不覆盖后来已完成的 Concern 2/4/5 正文论证。

## GitHub 仓库与固定版本

仓库地址：https://github.com/mathzhuangzou-cmyk/math31846-cooperative-kpp-chemotaxis

与返修稿对应的固定版本应使用核验后的完整提交号及其永久链接。当前没有创建 release tag 或 DOI。上传 GitHub 不等于已提交期刊。中文说明及英文替换文字见 docs/GITHUB_PUBLICATION_NOTES_CN.txt。

`docs/PACKAGE_VALIDATION.json` 保留本地 ZIP 打包时的历史核查结果，其中 `github_published: false` 对应上传前的状态。首次上传保留了科研源码和数值结果；本轮新增精度核查脚本并更新图1，原始与已核查的12份项目脚本及 PDE 表格数据保持不变。尚未选择软件许可证；公开可访问不等于已经授予开源使用许可。

审稿人要求代码随返修稿提供。建议公开 GitHub 仓库作为长期入口，同时把同一版本 ZIP 随返修稿作为补充材料提交；实际完成后才能写“已提供”。

## 第二位审稿人的补充修订

第一位审稿人的数值快照保留在提交 `6456d2b89d6cbebcb66ea2064128309524bf11fe`。
新增 `evidence/R2_benchmark_precision_check.py` 及其 JSON 输出，记录双精度、
Brent 求根区间 `[0.001,100]`、`xtol=1e-14` 和默认 `rtol=4*eps`，并用精确十进制参数进行60位精度复核。
连续临界点保留12位小数修正为 `mu_c=2.468635740033`；临界模态和阈值结论不变。
图1的图内标题已改为 modal-threshold curve，当前 PNG 和矢量 PDF 位于 `figures/`。
历史源脚本保留，当前图1请运行：

```bash
python evidence/R2_benchmark_precision_check.py --output rerun_outputs/reviewer2
```

新增依赖为 `mpmath==1.3.0`。该命令仅复核代数常数并绘制图1，不重跑 PDE 轨迹，
不改动任何已保存表格数据；求根容差不代表 PDE 离散误差界。

