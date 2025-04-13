from models.gan.GraphGAN.graph_gan import GraphGAN

if __name__ == "__main__":
    import os
    os.makedirs("GAN/cache", exist_ok=True)
    os.makedirs("GAN/results/link_prediction", exist_ok=True)
    graph_gan = GraphGAN()
    graph_gan.train()