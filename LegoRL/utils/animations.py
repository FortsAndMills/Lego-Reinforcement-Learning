def show_animation(frames):
    """
    generates animation inline notebook:
    input: frames - list of pictures
    """
    from matplotlib import pyplot as plt
    from matplotlib import animation, rc
    from IPython.display import HTML
    
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    return HTML(anim.to_jshtml())

# TODO: plot Q with the game
# def show_frames_and_distribution(frames, distributions, name, support):
#     """
#     generate animation inline notebook with distribtuions plot
#     input: frames - list of pictures
#     input: distributions - list of arrays of fixed size
#     input: name - title name
#     input: support - indexes for support of distribution
#     """ 
         
#     plt.figure(figsize=(frames[0].shape[1] / 34.0, frames[0].shape[0] / 72.0), dpi = 72)
#     plt.subplot(121)
#     patch = plt.imshow(frames[0])
#     plt.axis('off')
    
#     plt.subplot(122)
#     plt.title(name)
#     action_patches = []
#     for a in range(distributions.shape[1]):
#         action_patches.append(plt.bar(support, distributions[0][a], width=support[1]-support[0]))

#     def animate(i):
#         patch.set_data(frames[i])
        
#         for a, action_patch in enumerate(action_patches): 
#             for rect, yi in zip(action_patch, distributions[i][a]):
#                 rect.set_height(yi)

#     anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames) - 1, interval=50)
    
#     # not working in matplotlib 3.1
#     display(display_animation(anim, default_mode='loop'))