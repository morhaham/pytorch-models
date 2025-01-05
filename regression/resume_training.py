from utils import model_config,data_prepare, data_generate
from step_by_step import StepByStep

lr, model, optimizer, loss_fn = model_config()
new_sbs = StepByStep(model, loss_fn, optimizer)

print(f'preloaded model state: {model.state_dict()}\n')

new_sbs.load_checkpoint('model_checkpoint.pth')
print(f'loaded from checkpoint: {new_sbs.model.state_dict()}\n')

x, y = data_generate()
train_loader, val_loader = data_prepare(x, y)

new_sbs.set_loaders(train_loader, val_loader)
new_sbs.train(50)

print(model.state_dict())
# new_sbs.plot_losses()
