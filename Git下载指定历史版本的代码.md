# Git下载指定历史版本的代码

当你只想拿到项目的某个历史版本时，最常见的做法有 4 种：按提交哈希切换、按标签切换、按分支切换，以及直接从 GitHub 页面下载某个版本的压缩包。下面给出完整做法和适用场景。

## 1. 先拉取完整仓库

如果你还没有代码，先克隆仓库：

```bash
git clone git@github.com:用户名/仓库名.git
cd 仓库名
```

如果你已经有本地仓库，也可以先更新远程信息：

```bash
git fetch --all --tags
```

`git fetch` 不会修改你当前工作区，只会把远程的分支、提交和标签信息拉到本地。

## 2. 按某个历史提交下载代码

这是最精确的方式。先找到目标提交哈希：

```bash
git log --oneline --graph --decorate --all
```

假设你找到的提交哈希是 `a1b2c3d`，可以用下面几种方式：

### 方式 A：切换到该提交查看代码

```bash
git checkout a1b2c3d
```

此时仓库会进入“分离 HEAD”状态，适合临时查看或打包代码，但不适合直接继续开发。

### 方式 B：基于该提交新建分支

```bash
git checkout -b old-version a1b2c3d
```

这样你就得到一个基于历史版本的新分支，后续还可以继续修改、提交。

### 方式 C：只导出该版本代码到一个新目录

如果你只是想把某个版本导出成一个干净的文件夹：

```bash
git archive --format=zip --output=old-version.zip a1b2c3d
```

解压后就是该提交时的代码快照。

## 3. 按标签下载指定版本

如果项目作者给版本打了标签，通常更推荐用标签而不是提交哈希。

查看所有标签：

```bash
git tag
```

切换到指定标签：

```bash
git checkout v1.0.0
```

或者基于标签创建分支：

```bash
git checkout -b release-v1.0.0 v1.0.0
```

如果只想导出标签对应的代码：

```bash
git archive --format=zip --output=v1.0.0.zip v1.0.0
```

## 4. 按某个分支的历史状态下载

如果你知道是某个分支上的旧版本，可以先看分支历史：

```bash
git log --oneline main
```

然后找到对应提交，再用前面的方法切换或导出。

也可以直接创建一个从历史提交分出来的新分支：

```bash
git checkout -b my-old-copy <commit-hash>
```

## 5. 只下载指定提交的代码，不保留仓库历史

如果你的目标是“拿到代码就行”，不想保留 Git 历史，可以这样做：

```bash
git clone git@github.com:用户名/仓库名.git
cd 仓库名
git checkout <commit-hash>
```

然后把当前目录复制到别处，或者执行压缩打包。

如果仓库非常大，也可以考虑浅克隆，但浅克隆不适合随意切到很久以前的提交：

```bash
git clone --depth 1 git@github.com:用户名/仓库名.git
```

## 6. 从 GitHub 网页直接下载某个版本

如果代码托管在 GitHub，并且你只想快速拿一个版本，不想用命令行：

1. 打开仓库页面。
2. 切换到目标分支、标签或提交。
3. 点击 Code。
4. 选择 Download ZIP。

注意：网页下载通常拿到的是当前页面对应的快照，适合临时使用，不如 Git 命令灵活。

## 7. 常见场景对应方案

### 场景 1：我知道提交哈希

直接执行：

```bash
git checkout <commit-hash>
```

### 场景 2：我知道版本标签

直接执行：

```bash
git checkout <tag-name>
```

### 场景 3：我要继续在旧版本上开发

先创建分支：

```bash
git checkout -b old-dev <commit-hash>
```

### 场景 4：我只想导出一份干净代码包

使用：

```bash
git archive --format=zip --output=code.zip <commit-hash>
```

## 8. 恢复到最新版本

如果你切到了历史版本，想回到最新主分支：

```bash
git checkout main
git pull
```

如果你的主分支叫 `master`，把 `main` 换成 `master` 即可。

## 9. 实战建议

1. 优先用标签，其次用提交哈希。
2. 只查看旧代码时，用 `checkout` 即可。
3. 要继续改代码时，用 `checkout -b` 新建分支。
4. 如果只是交付一份代码包，用 `git archive` 或 GitHub 的 Download ZIP。

## 10. 最小命令总结

```bash
git fetch --all --tags
git log --oneline --graph --decorate --all
git checkout <commit-hash>
git checkout -b new-branch <commit-hash>
git archive --format=zip --output=code.zip <commit-hash>
```

如果你愿意，我也可以继续帮你把这份内容整理成“更适合发到 GitHub 的教程版”，或者直接补到你现有的 [上传GitHub流程.md](上传GitHub流程.md)。