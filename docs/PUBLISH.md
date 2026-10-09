# Publishing

The local Git repository contains code, figures, results and the original split. Model checkpoints, data and new training outputs are ignored. Before pushing, choose the GitHub account, repository name and visibility. No remote repository is configured by preparation.

After the repository is created, push the local main branch using your authenticated GitHub account. Upload `models/final_model.pt` separately as a release asset and add its release download link to `models/README.md`. A clone can demonstrate the recorded result and regenerate figures immediately; live prediction additionally requires the checkpoint.
