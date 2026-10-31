# -*- coding: utf-8 -*-
"""离线测试：一个字节都不打真的 AO3。

为什么要这样：AO3 是**按出口 IP** 封的。本机实测 418（真 Chrome 打开也一样），
从 Render 实测动态路径 525 —— 这种事封一次可能就不解了，所以「验证代码」不能拿
真站点当测试环境。

做法：客户端的所有网络请求都收在 `AO3Client._get` 一处，这里把它换成「按 URL 回
固定页面」，于是真实的参数拼装、BeautifulSoup 解析、缓存逻辑全都照跑，只是上游换成
了本地的假页面。要证明线上真实可达，那是 `/probe` 的活（跑在 Render 上，不是本机）。

    AO3Client._get  ←  假页面（本文件）
          ↓
    search / work_meta / download_text 的真实解析
"""
import asyncio
import json

import pytest

import ao3

SEARCH_PAGE = """<!doctype html><html><body>
<h3 class="heading">42 Found</h3>
<ol class="work index group">
<li class="work blurb group" role="article" id="work_4701869">
  <div class="header module">
    <h4 class="heading"><a href="/works/4701869">Test Work</a> by
      <a rel="author" href="/users/ann">Ann Writer</a></h4>
    <h5 class="fandoms heading"><a class="tag" href="/tags/hp">Harry Potter - J. K. Rowling</a></h5>
    <ul class="required-tags">
      <li><span class="rating general" title="General Audiences">G</span></li>
      <li><span class="category gen" title="Gen">Gen</span></li>
    </ul>
  </div>
  <ul class="tags commas">
    <li class="warnings"><a class="tag" href="#">No Archive Warnings Apply</a></li>
    <li class="relationships"><a class="tag" href="#">A/B</a></li>
    <li class="characters"><a class="tag" href="#">A</a></li>
    <li class="freeforms"><a class="tag" href="#">Fluff</a></li>
  </ul>
  <blockquote class="userstuff summary"><p>摘要正文。</p></blockquote>
  <p class="datetime">12 Oct 2026</p>
  <dl class="stats">
    <dd class="language">English</dd><dd class="words">1,234</dd><dd class="chapters">3/3</dd>
    <dd class="kudos">500</dd><dd class="bookmarks">12</dd><dd class="hits">9,000</dd>
  </dl>
</li>
</ol></body></html>"""

WORK_PAGE = """<!doctype html><html><body>
<h2 class="title heading">Test Work</h2>
<h3 class="byline heading"><a rel="author" href="/users/ann">Ann Writer</a></h3>
<dl class="tags">
  <dt>Rating:</dt><dd class="rating tags"><a class="tag" href="#">General Audiences</a></dd>
  <dt>Fandoms:</dt><dd class="fandom tags"><a class="tag" href="#">Harry Potter - J. K. Rowling</a></dd>
  <dt>Relationships:</dt><dd class="relationship tags"><a class="tag" href="#">A/B</a></dd>
  <dt>Characters:</dt><dd class="character tags"><a class="tag" href="#">A</a></dd>
  <dt>Additional Tags:</dt><dd class="freeform tags"><a class="tag" href="#">Fluff</a></dd>
</dl>
<div class="summary module"><blockquote class="userstuff"><p>摘要正文。</p></blockquote></div>
<dl class="stats">
  <dd class="language">English</dd><dd class="words">1,234</dd><dd class="chapters">3/3</dd>
  <dd class="kudos">500</dd><dd class="bookmarks">12</dd><dd class="hits">9,000</dd>
</dl></body></html>"""

DOWNLOAD_PAGE = """<!doctype html><html><body>
<div id="preface">
  <h1>Test Work</h1>
  <div class="byline">Ann Writer</div>
  <dl class="tags">
    <dt>Rating:</dt><dd>General Audiences</dd>
    <dt>Words:</dt><dd>1,234</dd>
  </dl>
</div>
<div id="chapters">
  <h2>Chapter 1: The Beginning</h2>
  <p>第一个自然段。</p>
  <p>第二个自然段。</p>
  <hr/>
  <h2>Chapter 2: The End</h2>
  <blockquote><p>引文段落。</p></blockquote>
</div></body></html>"""

PAGES = {
    "/works/search": SEARCH_PAGE,
    "/works/4701869": WORK_PAGE,
    "/downloads/4701869/work.html": DOWNLOAD_PAGE,
}


class FakeAO3:
    """替身：只换掉 _get（客户端唯一的网络出口），记下每次请求的 (url, params)。"""

    calls: list = []


async def _fake_get(self, url, params=None):
    FakeAO3.calls.append((url, params))
    for suffix, html in PAGES.items():
        if url.endswith(suffix):
            return html
    raise AssertionError(f"测试里没准备这个地址：{url}")


@pytest.fixture
def client(monkeypatch, tmp_path):
    FakeAO3.calls = []
    monkeypatch.setattr(ao3.AO3Client, "_get", _fake_get)   # 唯一的网络出口
    monkeypatch.setattr(ao3, "CACHE_DIR", tmp_path)         # 别写用户缓存目录
    c = ao3.AO3Client(min_interval=0.0)
    try:
        yield c
    finally:
        asyncio.run(c.close())


def run(coro):
    return asyncio.run(coro)


def test_search_turns_arguments_into_ao3_query_params(client):
    run(client.search(tags="Fluff", complete_only=True, sort_by="kudos",
                      categories="gen,m/m", page=2, pages=1))
    url, params = FakeAO3.calls[0]
    assert url.endswith("/works/search")
    pairs = dict(params)
    assert pairs["work_search[freeform_names]"] == "Fluff"
    assert pairs["work_search[complete]"] == "T"
    assert pairs["work_search[sort_column]"] == "kudos_count"  # AO3 的列名不是 "kudos"
    assert ("work_search[category_ids][]", "21") in params     # 逗号的多选要重复键
    assert ("work_search[category_ids][]", "23") in params
    assert ("page", "2") in params


def test_search_parses_a_blurb_into_a_work(client):
    res = run(client.search(tags="Fluff", pages=1))
    assert res["total_found"] == 42 and res["returned"] == 1
    w = res["works"][0]
    assert w["work_id"] == "4701869" and w["title"] == "Test Work"
    assert w["authors"] == ["Ann Writer"]
    assert w["fandoms"] == ["Harry Potter - J. K. Rowling"]
    assert w["kudos"] == 500 and w["words"] == 1234
    assert w["complete"] is True


def test_search_stops_asking_when_a_page_comes_back_empty(client):
    """最后一页之后别再白打 AO3 —— 每次多打一次都是给限流交学费。"""
    calls = []

    async def only_first_page(url, params=None):
        calls.append(params)
        return SEARCH_PAGE if len(calls) == 1 else "<html><body></body></html>"

    client._get = only_first_page
    res = run(client.search(tags="Fluff", pages=3))
    assert len(calls) == 2 and res["returned"] == 1


def test_work_meta_parses_tags_and_stats(client):
    meta = run(client.work_meta("4701869"))
    assert meta["title"] == "Test Work" and meta["authors"] == ["Ann Writer"]
    assert meta["rating"] == "General Audiences"
    assert meta["relationships"] == ["A/B"] and meta["tags"] == ["Fluff"]
    assert meta["kudos"] == 500 and meta["words"] == 1234
    assert meta["complete"] is True


def test_download_text_keeps_preface_and_body_apart(client):
    data = run(client.download_text("4701869"))
    assert data["title"] == "Test Work" and data["byline"] == "Ann Writer"
    assert data["meta"]["Rating"] == "General Audiences"
    assert "第一个自然段。" in data["text"] and "引文段落。" in data["text"]
    assert "Chapter 1: The Beginning" in data["text"]
    assert data["char_count"] == len(data["text"])


def test_second_read_of_the_same_fic_does_not_touch_ao3(client):
    run(client.download_text("4701869"))
    before = len(FakeAO3.calls)
    again = run(client.download_text("4701869"))
    assert len(FakeAO3.calls) == before, "正文缓存没生效，又去打了 AO3"
    assert again["title"] == "Test Work"


def test_a_bad_work_id_never_reaches_the_network(client):
    with pytest.raises(Exception):
        run(client.work_meta("../../etc/passwd"))
    assert FakeAO3.calls == []
