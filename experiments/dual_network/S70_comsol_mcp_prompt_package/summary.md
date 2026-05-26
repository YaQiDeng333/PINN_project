# S70 COMSOL MCP pilot prompt package

## 目的

S70 准备一个可直接交给 COMSOL MCP / COMSOL 相关对话的 pilot 任务提示文件，减少后续真实 COMSOL pilot 数据生成时的上下文整理成本。

## 新增产物

- `COMSOL_MCP_PILOT_PROMPT.md`

## 使用方式

将 `COMSOL_MCP_PILOT_PROMPT.md` 的内容交给 COMSOL MCP 项目或相关 COMSOL 对话，要求其生成 5-10 个真实 COMSOL-style multi-height Bz pilot samples，并输出 `signals_multiheight.csv`、`targets.npz`、`README.md`。

## 当前边界

S70 不调用 COMSOL，不生成真实 COMSOL 数据，不训练模型。它只是 handoff prompt package。

## 下一步建议

下一步应切到 COMSOL MCP 项目生成真实 pilot 数据。数据返回后，先用 S67 converter 转换，再用 S66 validator 验证，最后再决定是否运行 conditional runner。
