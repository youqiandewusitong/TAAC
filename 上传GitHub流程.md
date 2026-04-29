# 上传项目到 GitHub 流程

## 1. 初始化本地仓库
```bash
git init
```

## 2. 创建 .gitignore 文件
忽略不需要上传的文件（如 .ipynb_checkpoints、__pycache__ 等）

## 3. 添加文件并提交
```bash
git add .
git commit -m "Initial commit"
```

## 4. 配置 SSH 密钥（首次使用需要）
```bash
# 生成 SSH 密钥
ssh-keygen -t ed25519 -C "your_email@example.com"
# 一路回车即可

# 查看公钥
cat ~/.ssh/id_ed25519.pub
```

添加到 GitHub：
- 复制上面命令输出的公钥内容
- 访问 https://github.com/settings/keys
- 点击 "New SSH key"
- Title 随便填，Key 粘贴公钥
- 点击 "Add SSH key"
- 输入 GitHub 密码验证

## 5. 在 GitHub 创建仓库
- 访问 github.com
- 点击右上角 "+" → "New repository"
- 填写仓库名称
- **不要**勾选 "Initialize with README"
- 点击 "Create repository"

## 6. 连接远程仓库并推送
```bash
# 使用 SSH 地址（推荐）
git remote add origin git@github.com:你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

## 后续推送
```bash
git add .
git commit -m "更新说明"
git push
```

## 7. 如果 git push 提示 fetch first

如果你看到下面这种报错：

```bash
! [rejected] main -> main (fetch first)
error: failed to push some refs
```

说明远程仓库已经有你本地没有的提交，不能直接覆盖推送。正确做法是先把远程内容拉下来，再推送。

### 方案 A：推荐，先拉取再推送

```bash
git pull --rebase origin main
git push -u origin main
```

如果拉取时有冲突，先解决冲突，再执行：

```bash
git add .
git rebase --continue
git push -u origin main
```

### 方案 B：如果你确定要保留远程内容并合并

```bash
git pull origin main
git push -u origin main
```

这种方式会生成一次合并提交，适合你希望把远程已有内容也保留下来的情况。

### 方案 C：如果你明确要覆盖远程历史

只有在你非常确定远程内容可以被本地版本替换时才使用：

```bash
git push -f origin main
```

这个命令会强制覆盖远程分支，可能删除别人已经推送的提交，谨慎使用。

完成！
