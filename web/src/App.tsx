import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Cpu } from "lucide-react";

import { BuildInspector } from "./components/BuildInspector";
import { DraftWorkspace } from "./components/DraftWorkspace";
import { HeroPickerModal } from "./components/HeroPickerModal";
import { MatchSummaryBar } from "./components/MatchSummaryBar";
import { ModelInfoModal } from "./components/ModelInfoModal";
import {
  assignHeroToSlot,
  createEmptyDraft,
  createRandomDraft,
  parseDraftFromSearch,
  serializeDraftToSearch,
  swapSlotHeroes,
} from "./data/draftPresets";
import { loadOracleBundle, predictDraft, type LoadedOracleBundle } from "./ml/inference";
import type { DraftPrediction, DraftSlot, MatchPaceOption } from "./types/oracle";

function getInitialModelUrl(): string {
  if (typeof window === "undefined") {
    return "./models";
  }
  const params = new URLSearchParams(window.location.search);
  return params.get("model") || "./models";
}

export function App() {
  const [modelUrl, setModelUrl] = useState<string>(getInitialModelUrl);
  const [bundle, setBundle] = useState<LoadedOracleBundle | null>(null);
  const [loadingBundle, setLoadingBundle] = useState<boolean>(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [slots, setSlots] = useState<DraftSlot[]>(() => createEmptyDraft());
  const [matchPace, setMatchPace] = useState<MatchPaceOption>(0);
  const [pickerSlotIdx, setPickerSlotIdx] = useState<number | null>(null);
  const [modelInfoOpen, setModelInfoOpen] = useState<boolean>(false);

  const [prediction, setPrediction] = useState<DraftPrediction | null>(null);
  const [predicting, setPredicting] = useState<boolean>(false);
  const predictRunSeq = useRef(0);
  const bundleInitSeq = useRef(0);

  const runForecast = useCallback(
    async (
      activeBundle: LoadedOracleBundle,
      draftSlots: readonly DraftSlot[],
      paceMinutes: MatchPaceOption,
    ) => {
      if (draftSlots.some((s) => s.heroId === null)) {
        predictRunSeq.current += 1;
        setPrediction(null);
        setPredicting(false);
        return;
      }
      const seq = ++predictRunSeq.current;
      setPredicting(true);
      setLoadError(null);
      try {
        const result = await predictDraft(activeBundle, draftSlots, paceMinutes);
        if (predictRunSeq.current === seq) {
          setPrediction(result);
        }
      } catch (err) {
        if (predictRunSeq.current === seq) {
          setLoadError(
            err instanceof Error
              ? err.message
              : "Unable to compute match forecast in this browser.",
          );
        }
      } finally {
        if (predictRunSeq.current === seq) {
          setPredicting(false);
        }
      }
    },
    [],
  );

  const initBundle = useCallback(
    async (targetUrl: string, forceReload: boolean = false) => {
      const initSeq = ++bundleInitSeq.current;
      predictRunSeq.current += 1;
      setLoadingBundle(true);
      setPredicting(false);
      setLoadError(null);
      setPrediction(null);
      try {
        const loaded = await loadOracleBundle(targetUrl, forceReload);
        if (bundleInitSeq.current !== initSeq) {
          return;
        }
        setBundle(loaded);
        setModelUrl(loaded.baseUrl);

        const parsed =
          typeof window !== "undefined"
            ? parseDraftFromSearch(window.location.search, loaded.vocab.heroes.by_id)
            : null;

        const initialSlots = parsed ? parsed.slots : createEmptyDraft();
        const initialPace: MatchPaceOption = parsed ? parsed.matchPace : 0;

        setSlots(initialSlots);
        setMatchPace(initialPace);

        if (initialSlots.every((s) => s.heroId !== null)) {
          await runForecast(loaded, initialSlots, initialPace);
        }
      } catch (err) {
        if (bundleInitSeq.current === initSeq) {
          setLoadError(
            err instanceof Error
              ? err.message
              : "Unable to load active patch bundle.",
          );
        }
      } finally {
        if (bundleInitSeq.current === initSeq) {
          setLoadingBundle(false);
        }
      }
    },
    [runForecast],
  );

  useEffect(() => {
    void initBundle(modelUrl);
    // Run initial load once on mount; manual URL changes invoke initBundle directly.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const draftSearchQuery = useMemo(
    () => serializeDraftToSearch(slots, matchPace),
    [slots, matchPace],
  );

  useEffect(() => {
    if (typeof window === "undefined" || !bundle) return;
    if (slots.every((s) => s.heroId === null)) {
      const params = new URLSearchParams(window.location.search);
      if (params.has("r") || params.has("d") || params.has("pace") || params.has("pro")) {
        window.history.replaceState(null, "", window.location.pathname);
      }
      return;
    }
    window.history.replaceState(null, "", draftSearchQuery);
  }, [slots, bundle, draftSearchQuery]);

  const filledCount = useMemo(
    () => slots.filter((s) => s.heroId !== null).length,
    [slots],
  );

  const patchName = bundle?.vocab.game_version_name ?? "loading";

  const updateDraftAndMaybePredict = useCallback(
    (nextSlots: DraftSlot[], nextPace: MatchPaceOption) => {
      setSlots(nextSlots);
      setMatchPace(nextPace);
      if (bundle && nextSlots.every((s) => s.heroId !== null)) {
        void runForecast(bundle, nextSlots, nextPace);
      } else {
        predictRunSeq.current += 1;
        setPredicting(false);
        setPrediction(null);
      }
    },
    [bundle, runForecast],
  );

  const handleSelectHero = (heroId: number) => {
    if (pickerSlotIdx === null) return;
    const nextSlots = assignHeroToSlot(slots, pickerSlotIdx, heroId);
    setPickerSlotIdx(null);
    updateDraftAndMaybePredict(nextSlots, matchPace);
  };

  const handleDropSwap = (fromSlotIdx: number, toSlotIdx: number) => {
    if (fromSlotIdx === toSlotIdx) {
      return;
    }
    const swapped = swapSlotHeroes(slots, fromSlotIdx, toSlotIdx);
    updateDraftAndMaybePredict(swapped, matchPace);
  };

  const handleClearSlot = (slotIdx: number) => {
    const nextSlots = assignHeroToSlot(slots, slotIdx, null);
    updateDraftAndMaybePredict(nextSlots, matchPace);
  };

  const handleRandomDraft = () => {
    if (!bundle) return;
    const randomized = createRandomDraft(bundle.heroesList);
    updateDraftAndMaybePredict(randomized, matchPace);
  };

  const handleClearAll = () => {
    predictRunSeq.current += 1;
    setPredicting(false);
    setSlots(createEmptyDraft());
    setMatchPace(0);
    setPrediction(null);
    if (typeof window !== "undefined") {
      window.history.replaceState(null, "", window.location.pathname);
    }
  };

  const handleChangeMatchPace = (nextPace: MatchPaceOption) => {
    updateDraftAndMaybePredict(slots, nextPace);
  };

  const activePickerSlot =
    pickerSlotIdx !== null
      ? (slots.find((s) => s.slotIdx === pickerSlotIdx) ?? null)
      : null;

  return (
    <div className="app-shell">
      <header className="mast">
        <div className="container mast__inner">
          <div className="mast__brand">
            <a href="./" className="wordmark" aria-label="d2oracle home">
              <img
                src="./logo.png"
                alt=""
                width={28}
                height={28}
                className="wordmark__logo"
                aria-hidden="true"
              />
              <span>d2oracle</span>
            </a>
          </div>

          <div className="mast__meta">
            <button
              type="button"
              className={`status-chip status-chip--btn ${loadingBundle ? "is-loading" : "is-ready"}${modelInfoOpen ? " is-open" : ""}`}
              onClick={() => setModelInfoOpen(true)}
              disabled={bundle === null}
              aria-haspopup="dialog"
              aria-expanded={modelInfoOpen}
              title={bundle === null ? "Loading patch bundle" : "View active model details"}
            >
              <span className="status-chip__dot" aria-hidden="true" />
              <Cpu size={14} aria-hidden="true" />
              <span>Patch</span>
              <span className="measured">{patchName}</span>
              {bundle !== null && (
                <ChevronDown
                  size={13}
                  className="status-chip__caret"
                  aria-hidden="true"
                />
              )}
            </button>
          </div>
        </div>
      </header>

      <main className="container main-content">
        <DraftWorkspace
          slots={slots}
          heroesById={bundle?.vocab.heroes.by_id ?? {}}
          bundleReady={bundle !== null && !loadingBundle}
          loadingBundle={loadingBundle}
          predicting={predicting}
          filledCount={filledCount}
          draftQuery={draftSearchQuery}
          onOpenPicker={(slotIdx) => setPickerSlotIdx(slotIdx)}
          onDropSwap={handleDropSwap}
          onClearSlot={handleClearSlot}
          onRandomDraft={handleRandomDraft}
          onClearAll={handleClearAll}
        />

        <MatchSummaryBar
          prediction={prediction}
          loadingBundle={loadingBundle}
          predicting={predicting}
          error={loadError}
          matchPace={matchPace}
          onSelectMatchPace={handleChangeMatchPace}
        />

        <BuildInspector
          prediction={prediction}
          vocab={bundle?.vocab ?? null}
          matchPace={matchPace}
        />
      </main>

      <footer className="site-footer">
        <div className="container site-footer__inner">
          <p className="muted-note">
            Hero, ability, and item metadata from Valve Datafeed and dotaconstants.
          </p>
          <div className="site-footer__links">
            <a
              href="https://huggingface.co/datasets/rizeru/d2oracle-data"
              target="_blank"
              rel="noreferrer"
              className="site-footer__icon-link"
              aria-label="Hugging Face dataset"
              title="Hugging Face dataset"
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.65"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M5.5 11.5a6.5 6.5 0 1 1 13 0" />
                <circle cx="9.5" cy="9.6" r="0.85" fill="currentColor" stroke="none" />
                <circle cx="14.5" cy="9.6" r="0.85" fill="currentColor" stroke="none" />
                <path d="M9 12.2a3.2 3.2 0 0 0 6 0" />
                <path d="M4.2 13.6c-.8.4-1.2 1.4-.7 2.4.5 1.1 1.7 2.2 3.2 2.9 1 .5 1.9.4 2.4-.3.4-.6.1-1.5-.7-2l-1.5-.9c-.4-.3-.3-1.1.1-1.4" />
                <path d="M19.8 13.6c.8.4 1.2 1.4.7 2.4-.5 1.1-1.7 2.2-3.2 2.9-1 .5-1.9.4-2.4-.3-.4-.6-.1-1.5.7-2l1.5-.9c.4-.3.3-1.1-.1-1.4" />
              </svg>
            </a>
            <a
              href="https://github.com/ElRizeru/d2oracle"
              target="_blank"
              rel="noreferrer"
              className="site-footer__icon-link"
              aria-label="GitHub repository"
              title="GitHub repository"
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="currentColor"
                aria-hidden="true"
              >
                <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.55v-2.16c-3.2.69-3.87-1.36-3.87-1.36-.52-1.33-1.28-1.68-1.28-1.68-1.05-.72.08-.71.08-.71 1.16.08 1.77 1.19 1.77 1.19 1.03 1.76 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.55-.29-5.24-1.28-5.24-5.68 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18a11.04 11.04 0 0 1 5.8 0c2.21-1.49 3.18-1.18 3.18-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.41-2.69 5.39-5.25 5.67.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.51 11.51 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
              </svg>
            </a>
          </div>
        </div>
      </footer>

      <HeroPickerModal
        isOpen={pickerSlotIdx !== null}
        targetSlot={activePickerSlot}
        heroes={bundle?.heroesList ?? []}
        allSlots={slots}
        onSelectHero={handleSelectHero}
        onClose={() => setPickerSlotIdx(null)}
      />

      <ModelInfoModal
        isOpen={modelInfoOpen}
        vocab={bundle?.vocab ?? null}
        modelConfig={bundle?.modelConfig ?? null}
        onClose={() => setModelInfoOpen(false)}
      />
    </div>
  );
}
export default App;
