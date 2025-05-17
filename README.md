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

### 코드영역 태그 안에서 사용하기

태그에 markdown="1" 속성으로 jekyll 이 현재 태그 내부도 마크다운으로 파싱하도록 함
```md
<tag markdown="1">
  코드 영역 (``` 코드 ```)
</tag>
```

## history

[Guide](https://github.com/cotes2020/jekyll-theme-chirpy/wiki/Upgrade-Guide)

git 저장소 추가
```bash
git remote add chirpy https://github.com/cotes2020/chirpy-starter.git
```

## Google Search Console

### sitemap update
add : Gemfile => gem 'jekyll-sitemap'
```bash
$ bundle install
$ jekyll serve
```
https://junhyung96.github.io/sitemap.xml 로 접속 후
내용 복사 후 sitemap.xml (기존 혹은 생성)파일에 내용 업데이트

robots.txt 생성 및 아래 내용 추가
```
User-agent: *
Allow: /

Sitemap: https://junhyung96.github.io/sitemap.xml
```
google search console 사이트에서 sitemap 탭으로 이동
사이트맵 추가 및 제출
