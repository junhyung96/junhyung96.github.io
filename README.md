# Cresc.Dev

## 실행방법
```bash
$ bundle exec jekyll s
```

## 게시글 작성 방법
1. _posts 폴더에 YYYY-MM-DD-TITLE.EXTENSION 생성
2. Front Matter 작성
  ```yaml
    ---
    title: TITLE # 게시글 제목
    date: YYYY-MM-DD HH:MM:SS +/-TTTT # 포스트날짜 연-월-일 시-분-초 한국표준시+0900
    categories: [TOP_CATEGORIE, SUB_CATEGORIE]
    tags: [TAG]     # TAG names should always be lowercase
    ---
  ```
