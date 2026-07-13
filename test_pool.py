import torch
data = torch.load('C:/Users/mdiza/coding/KinshipForge-iz/pkl/pool_50samples.pkl', map_location='cpu', weights_only=False)
print('Type:', type(data))
if isinstance(data, dict):
    print('Keys:', len(data))
    for k in list(data.keys())[:5]:
        v = data[k]
        print('  {}: type={}, len={}'.format(k, type(v), len(v) if hasattr(v, '__len__') else 'N/A'))
        if isinstance(v, list) and len(v) > 0:
            print('    mu: {}, var: {}'.format(v[0][0].shape, v[0][1].shape))