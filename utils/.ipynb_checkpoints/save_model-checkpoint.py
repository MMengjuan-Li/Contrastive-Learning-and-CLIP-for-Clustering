import os
import torch
import shutil



def save_model(args, model, optimizer, scheduler, current_epoch, best_acc=None): 
 
    filename1 = f"checkpoint_best_1fc_clusterh.tar"
    out1 = os.path.join(args.model_path, filename1)
    # out = os.path.join(args.model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': current_epoch, 'best_acc': best_acc }
    torch.save(state, out1)

    if current_epoch % 100 == 0:
        filename2 = f"checkpoint_{current_epoch}_1fc_clusterh.tar"
        out2 = os.path.join(args.model_path, filename2)
        shutil.copy(out1, out2)  
