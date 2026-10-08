# Phone website setup (one time, about 15 minutes)

GitHub will run the model every **Thursday at 9:00 AM Eastern** and publish the Bet Builder as a
website. You can also start a run from your phone any time, for example for Sunday props.

## 1. Make the repository

1. Sign in at https://github.com (a free account works).
2. Top right **+** > **New repository**. Name it `nfl-edge` and choose **Public**. Free GitHub
   Pages needs a public repo; your API key stays secret (step 2). Click **Create repository**.
3. On the new repo page, click **uploading an existing file**.
4. Open your unzipped `nfl_edge` folder in File Explorer, select **everything inside it**
   (including the `.github` and `docs` folders), and drag it onto the page.
   Leave out `.venv`, `output`, and the `.parquet` and `games.csv` files in `data` if they're there.
5. Click **Commit changes**.

Check: the repo should list `.github`, `docs`, `edge`, `picks.py` and `requirements.txt`.
If `.github` is missing, Windows hid it. In File Explorer turn on **View > Show > Hidden items**
and drag the `.github` folder in on its own.

## 2. Add your odds key (kept secret)

Repo **Settings** > **Secrets and variables** > **Actions** > **New repository secret**
- Name: `ODDS_API_KEY`
- Secret: your key from the-odds-api.com

## 3. Turn on the website

Repo **Settings** > **Pages** > under **Build and deployment**, set **Source** to **GitHub Actions**.

## 4. Run it the first time

Repo **Actions** tab > **Update odds** (left side) > **Run workflow** > **Run workflow**.
The first run takes about 3 minutes because it downloads the NFL data. When it turns green, open:

    https://YOUR-GITHUB-NAME.github.io/nfl-edge/

Add it to your phone's home screen (Safari: Share > Add to Home Screen; Chrome: menu > Add to Home screen).

## 5. Move your bets over

On the website, open **My bets** > **Import bets** and pick `my_bets.json`.
Bets are saved in that browser. Use **Export bets** now and then as a backup, and import the
file on any other device you use.

## Every week

- **Thursday 9 AM:** updates by itself (game lines and props, about 99 credits).
- **Sunday morning (optional):** tap **Update odds now** in the site's banner, or in the GitHub
  app go to your repo > Actions > Update odds > Run workflow. To save credits, type teams into
  "Only these teams' games" (for example `IND,KC`), or untick props.
- You get 500 credits a month and a Thursday run uses about 99, so budget Sunday runs around that.
- GitHub sometimes starts scheduled runs 5–20 minutes late.

## If a run fails

Open the red run in the Actions tab and read the step that failed:
- "Add the ODDS_API_KEY secret": step 2 wasn't done, or the name is misspelled.
- "Odds API error 401": the key is wrong. Update the secret.
- "Odds API error 429" or a quota message: you're out of credits for the month.
