ssh-keygen -t ed25519 -C "xuanzhaopeng@gmail.com"
git config --global user.email "xuanzhaopeng@gmail.com"
git config --global user.name "Zhaopeng Xuan"

cat /root/.ssh/id_ed25519.pub
git clone git@github.com:xuanzhaopeng/curriculum0.git