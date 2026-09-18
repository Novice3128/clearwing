# cw-e2e 四路複審協議（條件觸發：判定翻案／對外開單前／發版判定）

歷史戰績（2026-09-17 深夜首戰）：單輪抓出 3 項實質錯誤——錯誤歸因（PR#43 回歸之說被 git 考古推翻）、
金鑰換行誤判（sha 不同≠值不同）、「逐位吻合」溢美（實差 0.16%＝一筆 audit 缺記）。
綠燈輪不審（期望值僅 P3 措辭，成本四個 agent session）。

派四個唯讀審查代理（並行），各自提示詞骨架如下（`<RUN_DIR>`＝受審 run 目錄）：

## R1 證據核實
> 你是獨立證據審查員（READ-ONLY；不得印任何金鑰值）。逐項核實 `<RUN_DIR>/report.md` 的每個量化宣稱
> 是否與原始產物一致：scenarios/*.summary.json＋*.frames.jsonl＋manifest.json＋ledger.json＋hud-proof.json＋
> state/pre-state.json（含 suite_sha256——審查對象版本錨定）。必查：秒數/幀數/statuses/工具數/成本/快取%/dup/late/對帳等式重算/帳本加總/
> 證據索引路徑存在性。另主動找清單外矛盾。輸出：[VERIFIED|MISMATCH|UNVERIFIABLE] 清單＋清單外新發現。

## R2 對抗性方法論
> 你是對抗性方法論審查員。攻擊判定：過度宣稱／單次樣本外推／比較基線斷裂／未揭露混淆因子
> （暖記憶、agent 自選工作量、「完成」定義）／判定詞彙與殘餘清單是否一一對應。讀 report.md＋
> frames .tools＋prompt 原文。輸出：各攻擊點 RISK＋成立與否＋具體修正措辭；最後判定表是否需要降級。

## R3 秘密與運維狀態
> 你是秘密紀律與運維審查員（絕不印金鑰值）。掃 `<RUN_DIR>` 全部檔案（grep api_key/authorization/Bearer/
> 32+hex；截圖 strings；以實際金鑰值反向 grep）。核對清理斷言（容器僅本輪 sid、埠釋放、/tmp、
> webui log api_key= 全 redact、8899 pid 未變、config 手術備份已刪、live health）。輸出 PASS/FAIL/WARN＋證據。

## R4 文件一致性＋根因考古
> 你是文件一致性與根因審查員。核對 report.md ↔ round-log/AGENTS.md/MEMORY 之間數字與事實；
> 對任何缺陷歸因做 git 考古驗證（git show/diff 只讀）——引用行號在 HEAD 上核實、「回歸」定性必須有
> 直接證據而非機制推論。輸出 [OK|ISSUE]＋修補字句；issue 草稿是否可直接開的結論。

修訂流程：彙整四路發現 → 修訂 report（rev2）→ 台帳同步。錯誤離開我方控制範圍之前是複審的最高價值時刻。
