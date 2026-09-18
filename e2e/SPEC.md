# cw-e2e SPEC — Clearwing 標準 E2E＋真實驗測套件

> 驗證方資產（放置於成員 repo 工作樹內、未追蹤）。成員請勿改動、勿 `git add` 本目錄。
> 版本：v1.1（2026-09-18；v1＝六輪實測＋四代理審查沉澱回溯標準化。v1.1 吸收 Codex PR r1 十五項發現：approval 佇列化、終幀排水窗、audit-present 硬門、cleanup 硬門、REGRESSION 判定、fallbacks YAML 結構注入、sessions JSON 解析、cost_cap/白名單/exit-code/報告標記判定、8899 進程新舊門、金鑰實值門；隨附 e2e/test_suite.py 離線回歸；r2 經三鏡 SubAgent 自審補強：busy-reject 自癒不計錯、排水窗純觀測守衛、surgery 失敗訊息去值化、sessions 503/paused 拒跑、金鑰門以進程 environ 為權威、值本位金鑰掃、selftest 子命令，回歸增至 24 項）。

## 1. 目的與驗收軸

以**五大使用者可觀測議題**為永續驗收軸（歷史判定見 round-log）：

| # | 軸 | 核心問題 |
|---|---|---|
| ① | 執行時間 | 全縱深滲透能否單趟、零人工干預完成（紀錄 warm/cold 標記） |
| ② | 報告 | 產出鏈路＋結構品質（findings/candidates 分區、match_quality、事實抽核） |
| ③ | 成本 | 定價閉環（audit tokens × PRICING ≡ meter）、快取命中、跨 session 隔離、HUD＝報告（**頁面渲染口徑**） |
| ④ | LLM 連線 | 混沌重試矩陣、fallback 鏈（能力驗證 vs 生產部署要分開陳述） |
| ⑤ | 模組整合 | 設定→模型鏈、工具組合實參傳遞、CVE 候選隔離、幀協議健康 |

## 2. 鐵則（違反即停止）

1. **目標白名單**：僅 `192.168.73.81/.82`（suite.yaml `targets`，runner 強制）。禁密碼噴灑/爆破（POLICY_DENY/DESTRUCTIVE_DENY 內建；歷史 M3 事件在案）。
2. **禁動**：`.venv/`、`pyproject.toml`、`Makefile`、`.gitignore`、成員 compose/:8080——**8899 活體用此 .venv 運行**。
3. **金鑰紀律**：key 檔路徑一律由 env（`CW_KEYFILE`＝webui key、`CW_LLM_PROFILE`＝LLM key yaml）注入，**無預設值**（歷史：文件化 key 曾失效、跨專案憑證路徑硬編碼＝事故級）。永不輸出值；log 出現 `api_key=` 立即 redact；金鑰檔比對必**去換行**（歷史誤判）。
4. **互斥**：`results/.lock`（O_EXCL＋陳舊鎖偵測：持有進程不在即接管）；Deep-fallback 需 `--approve-fallback`。
5. **8898 自起實例規格**：必經 `.venv/bin/clearwing webui` CLI（保證 api_key redact filter）；`--host 127.0.0.1`（`/api/sessions*`、`/api/metrics` **無鑰**）；env 隨機 key 自產自傳；cwd＝run 目錄（防污染 repo `results/`）；`CLEARWING_MCP_SERVERS_DIR` 指空目錄；pidfile 管理，**只 kill pidfile 的 pid，禁 `pkill -f clearwing`**（會同殺 8899/8080）；`setsid nohup … </dev/null`。
6. **config 手術（real-config，Deep-fallback）**：atomic write＋fsync＋SIGINT/TERM/EXIT trap 自動還原；重啟前查 `/api/sessions` 無 running；啟動時掃 `~/.clearwing/config.yaml.bak-*` 殘留（含金鑰）→ 有即拒跑；備份 0600＋時間戳＋測畢即刪；還原後 health＋diff＋pid 三驗。**CLEARWING_HOME 不隔離 provider config**（`~/.clearwing/config.yaml` 永遠疊加覆蓋同名鍵，config.py:146-161）——隔離實例測不到獨立拓撲，fallback 測試必須 real-config 手術＋8898 重啟承載。
7. **輸出隔離**：每 run 唯一 `results/<ts>-<tier>/`；寫檔前斷言目標不存在（歷史跨輪覆蓋教訓）；產出目錄名 `results/` 命中既有 gitignore（機制級防誤提交）。
8. **清理 SOP（每 run 收尾，斷言化）**：僅移除本輪記錄的 `clearwing-kali-<sid>` 容器（**禁 name-filter 全清**——docker 為全域共用）；`/tmp/report_*`/`kerbrute*` 淨；8787/8898 釋放；tmux 無（已自動斷言）；產物金鑰掃＝0（**值本位**：webui key＋LLM key 實值 byte-probe＋`api_key[=:?&]` 正則雙保險；截圖 strings＝0；`.key` 檔納掃；8898 自起實例 key 於 stop 時刪除）；8899 health 200 且 pid 未變。**成員 8899 的 webui log 掃描屬 R3 人工步驟**——live log redirect 目前寫入已刪除 inode 且 WS-accepted 行不經 redact filter（見 gh/unauth-endpoints-report.md 附錄），修復前自動掃不可行。

## 3. 層級與觸發（事件驅動，非每 PR 盲跑）

| 層 | 觸發 | 成本 | 內容摘要（細節=suite.yaml） |
|---|---|---|---|
| Tier0 verify | 每 run 內建 | $0 | 版本探針（**進程啟動時間 vs 產品碼 HEAD**＋cwd＋web-api.md commit，抓 pre-merge 舊碼）、health/401、金鑰等值（**以 8899 進程 environ 為權威**，鏡像檔僅回退；去換行；不等即硬門）、bak 殘留掃、埠佔用 abort、doctor/models 組態漂移、狀態＋.82 目標端快照 |
| Quick | 8899 HEAD 變更且 diff 觸及熱區（ui/web、llm、observability、reporting、agent/tools） | ≈$0.2–0.4 | zero-LLM 探針、fc 審批任務（status/重複幀/遲到幀）、**stop-frame**、**watchdog 低帽觸發**、warm×1（HUD 頁面渲染）、save_report 鏈、report -s（**內容標記判定**）、refuse×1（**注入數＋復原期望**）、裸 base_url 404 探針、定向 pytest（**exit code 判定**） |
| Full | 發版/里程碑 | ≈$3–5 | 五軸全量＋flag 基線（9 面）＋R3 抽核＋audit 缺記檢查＋混沌 4+2 式（含 429、deny 路徑；**每場景 gated：注入數＋complete=ok**）＋APPROVE_DELAY 變體＋baseline diff（±30% 帶） |
| Deep-cold | 排程/需要冷啟動數據 | ≈$3–5 | 隔離 home（**起跑斷言空記憶**＋首 recall 無回灌）；**cold-fulldepth 必為首場景**（任何預熱都會污染冷測）＋量測後煙測；與暖跑對照＝冷啟動係數 |
| Deep-fallback | **明確核准**（--approve-fallback） | ≈$0.1 | real-config 手術（**YAML 結構化注入 fallbacks 進 provider＋寫後自驗**；`/api/sessions` JSON 解析 running 才放行）＋8898 承載：全拒/回切/slow-fail 三場景（**slow-fail 的 complete=ok 門＝issue #57 驗收測試**） |
| Capability | 季度/發版 | 視凍結矩陣 | **凍結規劃，未實作**（comparison-test v3 Juice Shop ground-truth 能力基準＋S3 --inject-creds——需時另行建置） |

## 4. 判定門檻（三類；suite.yaml `thresholds`）

- **硬門（任一觸發＝FAIL）**：相鄰重複 agent_message 對>0；終態 complete 後遲到幀>0（**終幀後排水窗 `late_drain_s`=5s 內觀測**）；審批閉包破；**終態 complete 落在未執行審批之上（D1/#29 偵測器）**；快取命中=0 或 <80%（Full 大 session）；**對帳差>0.5% 或 audit 缺記 llm_call**；**有 metered cost_update 而 audit 檔缺席（audit-present）**；場景 `expect:` 全部鍵（approvals 下限/cancelled_turn/watchdog 觸發/complete_status/errors/**chaos_hits 注入數下限**/**graceful**）；flag faces 超 Full 基線（9）；**清理斷言失敗**（容器殘留/埠佔用/8899 pid 變/金鑰掃命中＝cleanup-\* 硬門）。`skipped` 場景不計 FAIL（severity=skip）。
- **比例門**：audit×PRICING 對帳 ≤0.5%；cache ≥80%（Full）；HUD≡報告（頁面渲染，精確到分）。
- **趨勢門**：時長/成本 vs 前次 Full ±30% 帶（首輪建立基線）；**失敗＝判定 REGRESSION（非 PASS）**。
- **判定詞彙**：PASS/FAIL/REGRESSION/SKIPPED＋每判定必附限定欄（n=、warm/cold、scope、口徑）。Quick PASS 僅授權「可併」＋免責聲明（n=1、協議面）；**發版＝Full 連續 2 次通過＋Deep-cold ≥1 樣本**。
- **驅動器協議紀律（approval）**：`approval_needed` 在 turn 內發射、turn 收尾才發 `complete(awaiting_approval)`，turn 活動期間伺服器**拒絕** `approve`（busy error 幀）——driver 一律**佇列決策、待 awaiting_approval 窗口開啟才沖刷**（殘餘微競速以 busy-reject 重試一次自癒）；終幀後不立即斷線（排水窗）。

## 5. 不變量優先（協議脆性對策）

契約文件 3 天改 10 次的現實下：driver 對欄位只做 feature-detect（`status or "ok"` 式），**斷言掛在不變量上**——(a) 對帳等式 audit tokens×PRICING ≡ meter；(b) 終態閉包（每 turn 必有 complete）；(c) 審批閉包。每 run 記錄 `git HEAD`＋`docs/web-api.md` commit。**紀律：協議斷言壞→優先懷疑產品改了**（歷史六輪每次「套件失效」實為產品缺陷）。

## 6. 棄用條款

套件連續兩次改動都只是追協議、零新發現 → 降級回手工流程（SPEC 保留，代碼凍結）。

## 7. 已知尖角（歷史教訓，操作必讀）

- 混沌 `--base-url` **必帶完整路徑**（裸 host→根路徑 `/chat/completions`→404 一擊斃命；issue 草稿 gh/base-url-path-drop.md）。
- per-session kali 容器不自動移除（收尾 docker rm，artifacts 留 `~/.clearwing/kali/<sid>/artifacts/`）。
- 成員重啟 8899 不帶 env key → 金鑰重生成、文件化 key 失效（verify 會抓）。
- audit 可能缺記 llm_call（歷史 0.16% 對帳差＝一筆缺記；audit-completeness 門會列缺記筆數）。
- **webui 瀏覽器路徑的 `?api_key=` query auth 會進 server access log 的 WS-accepted 行**（該行不受 redact filter 保護，且 filter 本身曾以 TypeError crash、log redirect 寫入已刪除 inode）——HUD 場景固有風險，已列入 gh/unauth-endpoints-report.md 附錄追蹤；風險窗內避免同時分發金鑰。
- playwright 依賴 `executable_path=/usr/bin/chromium`（browser extra 不在 install-dev、無瀏覽器 binary——兩個脆弱條件，勿 `playwright install`）。
- pytest 環境依賴失敗允許清單：`test_sandbox_integration.py`（Docker/msan/valgrind 類）；flake 協議＝隔離重跑一次再判。
- MCP servers dir 用 `Path.home()` 不隨 CLEARWING_HOME（8898 需 `CLEARWING_MCP_SERVERS_DIR` 指空）。
- 8898 與 8899 共用 repo 代碼：成員 git pull 瞬間兩實例版本同變——run 記錄兩者 HEAD。
- 進程新舊門的兩個邊界：**未提交的工作樹產品碼編輯**不受門管（僅比對 commit 時間）；**未來時間戳 commit**（時鐘偏移/rebase）會誤觸發——兩者皆有 `--force` 逃生口，理由寫進 run 記錄。
- surgery 相關：`/api/sessions` 非 200（含 503）一律**拒跑**（不能驗證≠放行）；`paused`（停在審批）視同 running 擋下；config 為 symlink 拒手術（atomic replace 會摧毀連結）；手術窗內成員編輯會被還原（ACTIVE 日誌已披露）。
- 驅動器排水窗（late_drain_s）為**純觀測**：終幀後不再發送任何幀（watchdog/stop/approve 全有守衛），遲到 cost_update 不會激發 stop。
- busy-reject（flush 後微競速）自癒重試一次且**不計入 error_count**（t1 以 errors==0 為門，自癒協議事件不得誤判 FAIL）。

## 8. 對成員的三個提案（gh/ 草稿，不代開）

1. approval_needed 結構化欄位（deny 判準掛結構化欄位、缺席 fail-closed——現行 deny 建在自然語言文案上，成員改一句文案防護即靜默失效）。
2. `docs/web-api.md` 契約版本頭（一行也好）。
3. 8899 無鑰端點面回報（`/api/sessions*`、`/api/metrics` 無 `require_api_key`；0.0.0.0 綁定下 LAN 可讀掃描結果）。

## 9. 四路複審（條件觸發：判定翻案／對外開單前／發版判定）

提示詞模板見 `REVIEW.md`：證據核實（逐數字重算）／對抗性方法論／秘密與運維狀態／文件一致性＋根因考古。歷史戰績：單輪抓出 3 項實質錯誤（錯誤歸因、金鑰換行誤判、溢美用語）。綠燈輪不審。


## 10. 協議變更維護清單（開發者版——產品改了什麼，套件要動哪裡）

> 設計原則：**不變量優先、欄位 feature-detect、未知幀只計數**（§5）。多數功能新增
> 零改動；以下是需要同步的完整清單。每次改動請跑 `cw-e2e run --tier quick` 回歸。

| 產品變動 | 套件動作 | 改動點 |
|---|---|---|
| 新幀類型（任何） | **零改動**——幀普查自動呈現（report「frame-type census」行）；新幀第一次出現即可見 | 無 |
| 新終態 status（ok/stopped/error 之外） | 同步兩處三態集合 | `runner.py ws_run` TRUST_STATUS 收線集合＋`analyze.py` terminal-closure |
| `cost_update` 語意變（如 session-scoped totals、#62） | 檢查 `ws_run` 的 `max(total_cost_usd)` 假設與 audit-completeness 對帳方向 | `runner.py` cost_update 分支＋`analyze.py evaluate` |
| `complete` 契約變（status/produced_new） | feature-detect 已容忍缺欄；語意變才動 D1 偵測器 | `runner.py` complete 分支 |
| LLM 新功能（思考等級/新參數） | **零改動**——驅動器不釘 ChatOptions；成本/快取比例門＋趨勢門會捕捉漂移 | 無（必要時調 `suite.yaml thresholds`） |
| 廠商改價 | 更新 pricing（唯一改點） | `suite.yaml pricing_*` 行 |
| 換模型/模型改名 | **零改動**——partial-fill 探針對拍 live config 的 provider.model（不硬編字串） | `--base-url` 路徑若換非 glm 模型才需傳 `--model` |
| 審批 payload 改（結構化欄位提案 gh/） | 採納後 deny 掛結構化欄位、缺席 fail-closed | `runner.py POLICY_DENY` 消費點 |
| 審批窗口語意改（如 mid-turn approve 開始被接受/排隊） | 沖刷觸發條件與 busy-retry 隨之調整 | `runner.py ws_run` 佇列/沖刷邏輯 |
| `web-api.md` 加版本頭（gh/ 草稿） | verify 增加版本比對、報告記版本 | `runner.py git_info` |
| 新端點/埠 | `suite.yaml webui` | 無代碼 |

**離線回歸**：改動套件任何檔案後先跑 `.venv/bin/python e2e/runner.py selftest`（=24 項離線測試，零 LLM 成本）再考慮 live 層。

**契約紀律**：套件斷言壞掉時優先懷疑產品改了（歷史六輪皆如此）；修套件前先跑
`git log -- docs/web-api.md` 對照。連續兩次維護只是追協議、零新發現 → 觸發 §6 棄用條款檢討。
