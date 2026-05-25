# =========================================
# SSH + GIT SETUP CHEATSHEET (GitHub)
# =========================================

# 1. Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# (Press ENTER for default path: /root/.ssh/id_ed25519)
# (Press ENTER for no passphrase for servers)

# -----------------------------------------

# 2. Show public key (COPY THIS)
cat /root/.ssh/id_ed25519.pub

# -----------------------------------------

# 3. Add to GitHub
# Go to:
# https://github.com/settings/keys
# Paste the public key

# -----------------------------------------

# 4. Test SSH connection
ssh -T git@github.com

# Expected:
# Hi username! You've successfully authenticated

# -----------------------------------------

# 5. Change repo to SSH
git remote set-url origin git@github.com:USERNAME/REPO.git

# -----------------------------------------

# 6. Verify remote
git remote -v

# -----------------------------------------

# 7. Push code
git push

# -----------------------------------------

# 8. If stuck (reset SSH agent issues)
unset GIT_ASKPASS
unset SSH_ASKPASS
export GIT_TERMINAL_PROMPT=1

# =========================================