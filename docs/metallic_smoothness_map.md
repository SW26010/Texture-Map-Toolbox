# Metallic / Smoothness 贴图工作流

## 当前状态

这个方向目前主要保留为历史 notebook 实验，不属于当前统一 CLI / API / GUI 主工作流。

当前现状是：

- 有历史 notebook：`metallic_smoothness_map.ipynb`
- 没有与 `luma` 对等的统一命令
- 没有稳定的 `api.metallic_*` 调用面
- 没有与 Qt Oklch 编辑器对等的 metallic/smoothness 专用 GUI

换句话说，这一页现在是“历史方向说明”，不是“当前产品入口文档”。

## 与当前主线的关系

仓库目前真正已经分层并持续维护的主线是 Oklch luma 工作流：

- notebook：`luma_color_map.ipynb`
- core：`texture_map_toolbox/core/luma.py`
- API：`texture_map_toolbox/api/luma.py`
- CLI：`texture_map_toolbox/cli/luma.py`
- GUI：`texture_map_toolbox/gui/qt_editor.py`

如果后续重新启动 metallic/smoothness 方向，建议直接沿用这条已经验证过的结构：

1. core 负责数值语义
2. API 提供稳定调用面
3. GUI 提供编辑与预览
4. CLI 只做参数解析和调度

## 原始目标

这条历史方向主要关注：

- 从现有贴图推断或组合 metallic / smoothness 信息
- 分离金属区域与非金属区域
- 提取高光、粗糙度或镜面相关结构
- 探索与 RGB、法线、AO 等其他贴图的联合使用

## 目前仍有的价值

虽然它不是当前主工作流，但这部分内容仍然有两类价值：

1. 保留材质贴图工具箱最早期的探索背景。
2. 为将来重启 PBR 类贴图工具提供方向参考。

更具体地说，当前 luma 主线已经验证了几件事可以直接复用到未来的 metallic/smoothness 工具：

- core / API / GUI / CLI 分层
- request / result JSON
- 可复用的预览与编辑器框架
- 统一的 smoke / 行为测试入口

## 建议阅读顺序

如果你的目标是当前仓库里最完整、最可复用、功能仍在持续演进的一条工作流，请优先阅读：

1. `README.md`
2. `docs/luma_color_map.md`
3. `docs/hsl_curve_editor_design.md`
4. `docs/cli.md`