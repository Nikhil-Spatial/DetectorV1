import bbox_visualizer as bbv

def draw_rectangles(image, coords, labels):
    image = image.permute(1, 2, 0).numpy()
    image = bbv.draw_multiple_rectangles(image, coords, bbox_color=(255, 0, 0),
                                         thickness=1)
    image = bbv.add_multiple_labels(image, labels, coords, size=0.5,
                                    thickness=1, draw_bg=False, text_color=
                                    (255, 255, 255))

    return image

def plot_history(epoch_list, history, hist_type: str):
    fig, ax = plt.subplots(1, figsize=(5, 5))

    ax.plot(epoch_list, history, c='k')
    ax.set_title(f"{hist_type} History")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(hist_type)

    plot_dir = Path(f"../outputs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(plot_dir / f"{hist_type}_history_plot.png")
    plt.close(fig)