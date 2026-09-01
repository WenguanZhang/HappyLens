# HappyLens Project Page

This directory contains the standalone HappyLens project introduction page. It is intentionally separate from the Sphinx documentation under `doc/`.

To preview it locally from the repository root:

```bash
python -m http.server 8080 --directory website
```

Then open `http://127.0.0.1:8080/`.

The page is dependency-free and can be deployed directly with GitHub Pages or any static hosting service. The prime- and zoom-lens cards open standalone workflow pages under `workflows/`; their full-resolution demonstration GIFs are stored under `assets/workflows/`.

The repository workflow at `.github/workflows/deploy-website.yml` publishes this
directory to GitHub Pages whenever `main` is updated. After the first push,
select **GitHub Actions** as the Pages source in the repository settings. The
Sphinx documentation is deployed separately through Read the Docs using the
repository-level `.readthedocs.yaml` configuration.
