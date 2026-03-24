# GABS Project

## Setup

```bash
git clone git@github.com:Grega2804/GABS-Project.git
cd GABS-Project
```

If you don't have SSH set up, use HTTPS instead:

```bash
git clone https://github.com/Grega2804/GABS-Project.git
```

## Workflow

**Before you start working**, always pull the latest changes:

```bash
git pull --rebase
```

**After making changes**, commit and push:

```bash
git add <files you changed>
git commit -m "Short description of what you did"
git push
```

## If your push is rejected

This means someone else pushed before you. Pull with rebase first:

```bash
git pull --rebase
git push
```

If there are conflicts, Git will tell you which files. Open them, fix the conflicts, then:

```bash
git add <fixed files>
git rebase --continue
git push
```

## Rules

- **Never use `git push --force`**
- Always use `git pull --rebase` (not just `git pull`)
- Commit often with clear messages
- Pull before you start working

## Useful commands

| Command | What it does |
|---|---|
| `git status` | See what files you changed |
| `git log --oneline` | See recent commits |
| `git diff` | See your uncommitted changes |
| `git pull --rebase` | Pull latest changes and rebase yours on top |
| `git reset HEAD~1` | Undo your last commit (keeps your changes) |
