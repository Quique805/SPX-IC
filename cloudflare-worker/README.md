# SpotGamma levels Worker

This Worker lets the dashboard save SpotGamma levels into GitHub without exposing a GitHub token in the browser.

## Required Worker variables/secrets

- `GITHUB_TOKEN`: fine-grained GitHub token with Contents read/write access to `Quique805/SPX-IC`
- `GITHUB_OWNER`: `Quique805`
- `GITHUB_REPO`: `SPX-IC`
- `GITHUB_BRANCH`: `main`
- `SPX_PIN`: private PIN typed in the dashboard
- `ALLOWED_ORIGIN`: dashboard origin, for example `https://quique805.github.io`

## Dashboard usage

1. Open the dashboard.
2. Go to `SpotGamma`.
3. Open `Configuracion de guardado remoto`.
4. Paste the Worker URL.
5. Type the private PIN.
6. Fill date, Call Wall, Put Wall, VT and Gamma Flip.
7. Press `Guardar nivel`.

The Worker updates `data/spotgamma-levels.json` and commits the change to GitHub. GitHub Actions will then use the new levels.
