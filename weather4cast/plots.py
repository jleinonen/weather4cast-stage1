import os

from matplotlib import colors, gridspec, pyplot as plt


def plot_frame(img):
    plt.imshow(img, norm=colors.Normalize(0,1))
    plt.gca().tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)
    

def plot_sequence(model, batch_gen, batch_index=0, var_index=0,
    sample=0, plot_fn="../figures/prediction_attp.pdf", rounded=False,
    past_frames=(1,3), future_frames=(0,2,7,15,31), colorbar=False):

    batch = batch_gen[batch_index]
    pred = model.predict(batch[0])

    plt.figure(figsize=(17,5))
    gs = gridspec.GridSpec(8,len(past_frames)+len(future_frames),
        wspace=0.05)

    for i in range(len(past_frames)):
        plt.subplot(gs[2:6,i])
        frame = past_frames[i]
        plot_frame(batch[0][sample,frame,:,:,var_index])
        plt.title("T={}".format(-batch[0].shape[1]+frame+1))

    if colorbar:
        cax = plt.gcf().add_axes([0.14, 0.2, 0.18, 0.05])
        plt.colorbar(cax=cax, orientation='horizontal')

    for i in range(len(future_frames)):
        plt.subplot(gs[0:4,i+len(past_frames)])
        frame = future_frames[i]
        plot_frame(batch[1][0][sample,frame,:,:,0])
        plt.title("T=+{}".format(frame+1))
    plt.ylabel("Real", fontsize=14)
    plt.gca().yaxis.set_label_position("right")

    for i in range(len(future_frames)):
        plt.subplot(gs[4:8,i+len(past_frames)])
        frame = future_frames[i]
        plot_frame(pred[sample,frame,:,:,0])
        if rounded:
            plt.contour(pred[sample,frame,:,:,0], levels=[0.5], colors='w')
    plt.ylabel("Predicted", fontsize=14)
    plt.gca().yaxis.set_label_position("right")

    plt.savefig(plot_fn, bbox_inches='tight')
    plt.close('all')
