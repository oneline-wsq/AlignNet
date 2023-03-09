
def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'vgg'       : 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50'  : 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth'
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)