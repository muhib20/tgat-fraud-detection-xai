# Models

`tgat_trained.pt` — TGAT weights (seed 0) used to produce every result in
`results/`. Load it rather than retraining if you only want to reproduce the
SHAP and GNNExplainer outputs.

```python
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'tgat_trained.pt')))
```
