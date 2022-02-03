This is just me following along the Huggingface ML tutorial, plus Kyle's ML workshop.
* https://huggingface.co/course/chapter1/3?fw=pt
* https://docs.google.com/document/d/1SPr8lIi_TwdXkPOJ-K55EMZwqn3BNzDVBrqdGHNYuTo/edit
* https://docs.google.com/document/d/1RMxsqwN-Ylo7keG7yvPDMq5BLO06wroXlFEOxX4awbY/edit

To access my EC2 instance:
* from .ssh/
* `ssh -i "s2dev.pem" ubuntu@ec2-35-88-163-180.us-west-2.compute.amazonaws.com`
* `source activate pytorch_latest_p37`
* There's a bunch of files in there, and IDK how to edit the files there unless it's Nano/Vim.  
  * Which is annoying so that's why I'm just editing the files on my local playML repo, then `git push origin -f`.
  * Then, in the terminal where I have the SSH'd instance, `git pull --rebase origin main`, then run the files I want.
  * I guess there's a better way to do it? via Notebook or soemthing? IDK
    * Is this it? https://gist.github.com/jakechen/faf0500132d46d83517004bbfedbe5de#open-ssh-tunnel-and-start-jupyter-notebook
    * I coudln't get it to work
