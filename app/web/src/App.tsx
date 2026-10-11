import { useEffect, useRef } from "react";
import { START_COMMAND } from "./api";
import { useBackend, type Connection } from "./backend";
import { AboutPage } from "./components/AboutPage";
import { ClassifyPage } from "./components/ClassifyPage";
import { MonitorPage } from "./components/MonitorPage";
import { ResultsPage } from "./components/ResultsPage";
import { BottomNav, Sidebar, Topbar } from "./components/Shell";
import { EmptyState, PageHeader } from "./components/ui";
import { PAGES, useRoute } from "./route";
import { ThemeContext, useTheme } from "./theme";

function BackendState({
  title,
  connection,
  error,
  unreachable,
}: {
  title: string;
  connection: Connection;
  error: string | null;
  unreachable: boolean;
}) {
  return (
    <div className="page">
      <PageHeader title={title} />
      {connection === "connecting" ? (
        <EmptyState busy title="Connecting to the backend">
          Requesting the model list from the API.
        </EmptyState>
      ) : (
        <EmptyState icon="alert" title="The backend is not reachable">
          {unreachable ? (
            <>
              <p>The API did not respond. Start it from the project root and this page reconnects on its own.</p>
              <pre className="codeblock">{START_COMMAND}</pre>
            </>
          ) : (
            <p>{error}</p>
          )}
        </EmptyState>
      )}
    </div>
  );
}

export default function App() {
  const tab = useRoute();
  const [theme, toggleTheme] = useTheme();
  const { models, families, connection, error, unreachable } = useBackend();
  const main = useRef<HTMLElement>(null);
  const firstRender = useRef(true);
  const page = PAGES.find((p) => p.key === tab) ?? PAGES[0];

  // On a page change: set the document title, return to the top and move focus to the page for screen readers.
  useEffect(() => {
    document.title = `${page.label} | DDoS Detection`;
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    window.scrollTo(0, 0);
    main.current?.focus({ preventScroll: true });
  }, [tab, page.label]);

  return (
    <ThemeContext.Provider value={theme}>
      <a
        className="skip-link"
        href="#main"
        onClick={(event) => {
          event.preventDefault();
          main.current?.focus();
        }}
      >
        Skip to content
      </a>
      <div className="app">
        <Sidebar tab={tab} connection={connection} models={models} theme={theme} onToggleTheme={toggleTheme} />
        <div className="app__body">
          <Topbar connection={connection} theme={theme} onToggleTheme={toggleTheme} />
          <main id="main" ref={main} tabIndex={-1} className="main">
            {models ? (
              <>
                <div hidden={tab !== "monitor"}>
                  <MonitorPage models={models} active={tab === "monitor"} />
                </div>
                <div hidden={tab !== "classify"}>
                  <ClassifyPage models={models} families={families} active={tab === "classify"} />
                </div>
                {tab === "results" && <ResultsPage models={models} />}
                {tab === "about" && <AboutPage />}
              </>
            ) : tab === "about" ? (
              <AboutPage />
            ) : (
              <BackendState title={page.label} connection={connection} error={error} unreachable={unreachable} />
            )}
          </main>
        </div>
        <BottomNav tab={tab} />
      </div>
    </ThemeContext.Provider>
  );
}
