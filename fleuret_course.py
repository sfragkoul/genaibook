import torch


z_dim = 8
nb_hidden = 100

model_G = nn.Sequential(nn.Linear(z_dim, nb_hidden),
nn.ReLU(),
nn.Linear(nb_hidden, 2))

model_D = nn.Sequential(nn.Linear(2, nb_hidden),
nn.ReLU(),
nn.Linear(nb_hidden, 1),
nn.Sigmoid())

batch_size, lr = 10, 1e-3
optimizer_G = optim.Adam(model_G.parameters(), lr = lr)
optimizer_D = optim.Adam(model_D.parameters(), lr = lr)


for e in range(nb_epochs):
  for t, real_batch in enumerate(real_samples.split(batch_size)):
    z = real_batch.new(real_batch.size(0), z_dim).normal_()
    fake_batch = model_G(z)
    D_scores_on_real = model_D(real_batch)
    D_scores_on_fake = model_D(fake_batch)
    if t%2 == 0:
      loss = (1 - D_scores_on_fake).log().mean()
      optimizer_G.zero_grad()
      loss.backward()
      optimizer_G.step()
    else:
      loss = - (1 - D_scores_on_fake).log().mean() \
      - D_scores_on_real.log().mean()
      optimizer_D.zero_grad()
      loss.backward()
      optimizer_D.step()
