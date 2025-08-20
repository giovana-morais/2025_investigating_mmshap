from matplotlib.colors import rgb2hex
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def visualize_shapley_analysis(text_shapley_values, question_tokens,
                             audio_signal, audio_shapley_values, sample_rate,
                             gt_start, gt_end,
                             max_abs_value=None, colormap='viridis',
                             figsize=(10, 8), idx=None, answer_tokens=None,
                             threshold=0.8, save=True):
    """
    Combined visualization of text and audio Shapley values with shared color scaling.
    """
    # Determine global color scaling
    # if max_abs_value is None:
    #     text_max = np.max(np.abs(text_shapley_values))
    #     audio_max = np.max(np.abs(audio_shapley_values))
    #     max_abs_value = max(text_max, audio_max)

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 2, height_ratios=[2,4,1,1,1], width_ratios=[10,1])

    if idx is not None:
        text_shapley_values = text_shapley_values[:, idx]
        audio_shapley_values = audio_shapley_values[:, idx]
    else:
        text_shapley_values = text_shapley_values.sum(axis=1)
        audio_shapley_values = audio_shapley_values.sum(axis=1)

    visualize_text(fig, gs, question_tokens, text_shapley_values,
            colormap=colormap,
            intensity_threshold=threshold)

    visualize_audio(fig, gs, audio_signal, sample_rate, audio_shapley_values,
            gt_start=gt_start, gt_end=gt_end, colormap=colormap,
            intensity_threshold=threshold)


    # --- Formatting ---
    # title = plt.suptitle(highlight_title(answer_tokens, idx), y=0.98, fontsize=14)
    if save:
        if idx is None:
            plt.savefig("aggregated_output.pdf", format="pdf", bbox_inches="tight")
        else:
            plt.savefig(f"{idx}_{answer_tokens[idx].strip()}.pdf", format="pdf", bbox_inches="tight")
    # print(title)
    # title(fig, answer_tokens, idx)
    # plt.tight_layout()

    plt.subplots_adjust(right=0.9, hspace=0.15)
    plt.show()

    return

def highlight_title(answer_tokens, idx):
    title = "Model response: "
    for i, t in enumerate(answer_tokens):
        to_add = t
        if i == idx:
            to_add = f">{t.strip()}<"

        title += to_add + " "

    return title

def visualize_audio(fig, gs, audio_signal, sample_rate, audio_shapley_values,
        gt_start, gt_end,
        intensity_threshold, colormap):

    # --- Time axis setup ---
    total_duration = len(audio_signal) / sample_rate  # Reused for all plots
    time_axis = np.linspace(0, total_duration, len(audio_signal))
    shapley_time_axis = np.linspace(0, total_duration, len(audio_shapley_values))

    # --- 1. Signal plot (top subplot) ---
    ax_signal = fig.add_subplot(gs[1, 0])
    ax_signal.plot(time_axis, audio_signal, color='gray', alpha=0.7, linewidth=0.5)
    ax_signal.set_yticks([])

    # Add ground truth rectangle
    if gt_start is not None and gt_end is not None:
        ymin, ymax = ax_signal.get_ylim()
        ax_signal.axvspan(gt_start, gt_end, ymin=0, ymax=1,
                         color='red', alpha=0.3, label='Ground Truth')
        ax_signal.legend(loc='upper right')
        ax_signal.tick_params(axis='x', bottom=False, labelbottom=False)

    # --- Audio Shapley visualizations ---
    # Calculate Shapley value components
    abs_shapley = np.abs(audio_shapley_values)
    pos_shapley = np.clip(audio_shapley_values, a_min=0, a_max=None)
    neg_shapley = np.clip(audio_shapley_values, a_min=None, a_max=0)

    # FIXME: does this make sense?
    # if idx is None:
    #     abs_shapley = abs_shapley.sum(axis=1)
    #     pos_shapley = pos_shapley.sum(axis=1)
    #     neg_shapley = neg_shapley.sum(axis=1)

    max_abs_value = np.max(abs_shapley)

    # 2. Absolute Shapley values (heatmap)
    ax_abs = fig.add_subplot(gs[2, 0], sharex=ax_signal)  # Share x-axis with signal plot
    im_abs = ax_abs.imshow(
        np.repeat(abs_shapley.reshape(1, -1), 10, axis=0),
        aspect='auto',
        cmap=colormap,
        extent=[0, total_duration, 0, 1],  # Match audio signal's time range
        vmin=0,
        vmax=max_abs_value
    )
    ax_abs.set_ylabel('Absolute\nValue', rotation=0, ha='right', va='center', fontsize=10)
    ax_abs.set_yticks([])
    ax_abs.tick_params(axis='x', bottom=False, labelbottom=False)
    # ax_abs.set_xticks([])

    # 3. Positive Shapley values (heatmap)
    ax_pos = fig.add_subplot(gs[3, 0], sharex=ax_signal)  # Share x-axis
    im_pos = ax_pos.imshow(
        np.repeat(pos_shapley.reshape(1, -1), 10, axis=0),
        aspect='auto',
        cmap=colormap,
        extent=[0, total_duration, 0, 1],
        vmin=0,
        vmax=max_abs_value
    )
    ax_pos.set_ylabel('Positive\nOnly', rotation=0, ha='right', va='center', fontsize=10)
    ax_pos.set_yticks([])
    ax_pos.tick_params(axis='x', bottom=False, labelbottom=False)
    # ax_pos.set_xticks([])

    # 4. Negative Shapley values (heatmap)
    ax_neg = fig.add_subplot(gs[4, 0], sharex=ax_signal)  # Share x-axis
    im_neg = ax_neg.imshow(
        np.repeat(np.abs(neg_shapley).reshape(1, -1), 10, axis=0),
        aspect='auto',
        cmap=colormap,
        extent=[0, total_duration, 0, 1],
        vmin=0,
        vmax=max_abs_value
    )
    ax_neg.set_ylabel('Negative\nOnly', rotation=0, ha='right', va='center', fontsize=10)
    ax_neg.set_yticks([])
    ax_neg.set_xlabel('Time (seconds)', fontsize=12)

    # --- Sync x-axis limits ---
    ax_signal.set_xlim(0, total_duration)  # Force all plots to match

    # --- Colorbar ---
    cax = fig.add_subplot(gs[1:, 1])
    norm = mpl.colors.Normalize(vmin=0, vmax=max_abs_value)
    sm = cm.ScalarMappable(norm=norm, cmap=colormap)
    fig.colorbar(sm, cax=cax, label='Shapley Value Magnitude')

    for ax in [ax_signal, ax_abs, ax_pos, ax_neg, cax]:
        ax.set_frame_on(False)

    return

def visualize_text(fig, gs, question_tokens, question_shapley_values, intensity_threshold, colormap):
    # use first row, all columns
    question_text = fig.add_subplot(gs[0,0:])
    question_text.set_xlim(0, 1)  # set fixed x limits
    question_text.set_ylim(0, 1)  # set fixed y limits
    question_text.axis('off')     # remove ugly box

    # Initial position
    x_pos = 0.0
    y_pos = 0.98
    line_height = 0.16 # Space between lines
    previous_text_obj = None  # Track the previous text object

    threshold = intensity_threshold*np.max(np.abs(question_shapley_values))
    cmap = mpl.colormaps[colormap]

    for t, v in zip(question_tokens, question_shapley_values):
        # leave inside the loop to restart the settings every token
        font_settings = {
            "fontname": "monospace",
            "fontsize": 14
        }

        # Set custom properties
        intensity = abs(v)
        rgba = cmap(intensity)
        highlight_props = None

        if intensity > threshold:
            highlight_props = {'facecolor': rgba, 'pad': 0.05, "edgecolor":
            "none", "boxstyle":"round"}
            if colormap == "binary":
                font_settings["color"] = "white"

        # Create text object with current properties
        text_obj = question_text.text(x_pos, y_pos, t.strip(),
                fontdict=font_settings,
                                  verticalalignment='top',
                                  horizontalalignment='left',
                                  wrap=True,
                                  bbox=highlight_props)

        # Get the bounding box of the text to determine its width
        renderer = fig.canvas.get_renderer()
        bbox = text_obj.get_window_extent(renderer=renderer)
        bbox_inches = bbox.transformed(question_text.transData.inverted())
        text_width = bbox_inches.width

        # Update position for next word
        x_pos += text_width + 0.005  # Add small padding

        # If we're going to exceed the subplot width, move to next line
        if x_pos > 0.95:  # Leave 10% margin on right
            x_pos = 0
            y_pos -= line_height

    return
