# Publication notes

Repository name: **mini-vla-maze**

Description:

> Language-conditioned visual pushing: compact imitation learning, explicit planning, and reproducible obstacle benchmarks.

Suggested topics: `robot-learning`, `imitation-learning`, `visual-grounding`, `pytorch`, `planning`, `simulation`.

No remote repository has been created and nothing has been published to an account.

Create an empty GitHub repository, then run the following **inside a separate extracted copy of this project**, substituting the repository URL:

```bash
git init -b main
git add .
git commit -m "Add visual pushing experiments and reproducible benchmarks"
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

The README uses relative media links, so its images and GIFs work after the files are pushed together. `gallery.html` can be opened locally; GitHub's file browser shows its source. Hosting it as a website would be a separate optional step.

Reference weights are small enough for ordinary Git. Generated training datasets are excluded under `runs/`. The portable notebook embeds the checkpoint intentionally so Colab needs only one upload.
