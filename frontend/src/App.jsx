import { useEffect, useMemo, useRef, useState } from "react";
import {
  BookOpen,
  CheckCircle2,
  CircleAlert,
  Copy,
  Eraser,
  ExternalLink,
  Loader2,
  MessageSquareText,
  Moon,
  Send,
  Settings2,
  Sun,
} from "lucide-react";
import { askQuestion, fetchStatus } from "./api";
import CampusScene3D from "./CampusScene3D.jsx";

const QUICK_PROMPTS = [
  "What is the admission process at CUK?",
  "Show the latest important notices.",
  "Who is the Dean of School of Media Studies?",
  "Give me contact details for the university office.",
];

const ANSWER_STYLES = [
  { id: "balanced", label: "Balanced" },
  { id: "concise", label: "Concise" },
  { id: "detailed", label: "Detailed" },
];
const CUK_LOGO = "/CUKLogo.png";

function formatChunkCount(value) {
  const count = Number(value || 0);
  return `${count.toLocaleString()} ${count === 1 ? "chunk" : "chunks"}`;
}

function formatTime(date) {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function buildHistory(messages) {
  const history = [];
  for (let index = 0; index < messages.length - 1; index += 1) {
    const user = messages[index];
    const bot = messages[index + 1];
    if (user.role === "user" && bot?.role === "assistant") {
      history.push({ user: user.content, bot: bot.content });
      index += 1;
    }
  }
  return history.slice(-6);
}

function InlineText({ text }) {
  const tokens = [];
  const pattern = /(\*\*[^*]+\*\*|\[[^\]]+\]\(https?:\/\/[^)\s]+\)|https?:\/\/[^\s<]+)/g;
  const value = String(text);
  let lastIndex = 0;
  let match;

  while ((match = pattern.exec(value)) !== null) {
    if (match.index > lastIndex) {
      tokens.push({ type: "text", value: value.slice(lastIndex, match.index) });
    }

    tokens.push({ type: "token", value: match[0] });
    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < value.length) {
    tokens.push({ type: "text", value: value.slice(lastIndex) });
  }

  return tokens.map((token, index) => {
    const key = `${token.value}-${index}`;
    if (token.type === "text") {
      return <span key={key}>{token.value}</span>;
    }

    if (token.value.startsWith("**") && token.value.endsWith("**")) {
      return <strong key={key}>{token.value.slice(2, -2)}</strong>;
    }

    const markdownLink = token.value.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/);
    if (markdownLink) {
      return (
        <a key={key} href={markdownLink[2]} target="_blank" rel="noreferrer">
          {markdownLink[1]}
        </a>
      );
    }

    const trailing = token.value.match(/[).,;:!?]+$/)?.[0] || "";
    const href = trailing ? token.value.slice(0, -trailing.length) : token.value;
    return (
      <span key={key}>
        <a href={href} target="_blank" rel="noreferrer">
          {href}
        </a>
        {trailing}
      </span>
    );
  });
}

function isPipeTableRow(line) {
  const trimmed = line.trim();
  return trimmed.includes("|") && (trimmed.match(/\|/g) || []).length >= 2;
}

function isSeparatorRow(cells) {
  return cells.every((cell) => /^:?-{2,}:?$/.test(cell.trim()) || cell.trim() === "");
}

function splitTableCells(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function normalizeDisplayText(content) {
  return String(content || "")
    .replace(/\s*¢\s*/g, "\n")
    .replace(/\s+\+\s+(?=Semester\b)/gi, "\n")
    .replace(/\)\s+(?=[A-Z]{2,}\.\d{2}\.\d{3}\s)/g, ")\n")
    .replace(/(\d{1,3}\s+ESE)\s+(?=[A-Z]{2,}\.\d{2}\.\d{3}\s)/gi, "$1\n");
}

function cleanHeadingText(text) {
  return String(text || "")
    .replace(/\s*[=-]{4,}\s*$/g, "")
    .replace(/\s+#+\s*$/g, "")
    .trim();
}

function extractSemester(line) {
  const match = line.match(/^(?:[-+*]\s*)?Semester\s+([IVX]+|\d+)\s*:?\s*(.*)$/i);
  if (!match) return null;
  return {
    semester: `Semester ${match[1].toUpperCase()}`,
    rest: match[2].trim(),
  };
}

function parseCourseDetail(line, semester) {
  const cleaned = line
    .replace(/^(?:[-+*]\s*)/, "")
    .replace(/\s+\[[^\]]+\]\s*$/, "")
    .trim();
  if (!semester || !cleaned) return null;

  const match = cleaned.match(
    /^(?:(?<code>[A-Z]{2,}(?:\.\d{2})?\.\d{3}[A-Z]?|[A-Z]{2,}\s?\d{3}\s?L?)\s+)?(?<title>.+?)\s*\((?<credits>\d{1,2})\s*credits?,\s*(?<cia>\d{1,3})\s*CIA,\s*(?<ese>\d{1,3})\s*ESE\)$/i,
  );
  if (!match?.groups) return null;

  return {
    semester,
    code: match.groups.code || "",
    title: match.groups.title.replace(/\s+/g, " ").trim(),
    credits: match.groups.credits,
    cia: match.groups.cia,
    ese: match.groups.ese,
  };
}

function TableBlock({ rows }) {
  const parsed = rows.map(splitTableCells).filter((cells) => cells.some(Boolean));
  if (!parsed.length) return null;

  const separatorIndex = parsed.findIndex(isSeparatorRow);
  const header =
    separatorIndex > 0
      ? parsed[separatorIndex - 1]
      : parsed.length > 1
        ? parsed[0]
        : null;
  const body = parsed.filter((cells, index) => {
    if (isSeparatorRow(cells)) return false;
    if (separatorIndex > 0 && index === separatorIndex - 1) return false;
    if (separatorIndex === -1 && parsed.length > 1 && index === 0) return false;
    return true;
  });
  const columnCount = Math.max(...parsed.map((cells) => cells.length));

  function pad(cells) {
    return Array.from({ length: columnCount }, (_, index) => cells[index] || "");
  }

  return (
    <div className="table-scroll">
      <table className="answer-table">
        {header && (
          <thead>
            <tr>
              {pad(header).map((cell, index) => (
                <th key={`${cell}-${index}`}>
                  <InlineText text={cell} />
                </th>
              ))}
            </tr>
          </thead>
        )}
        <tbody>
          {body.map((row, rowIndex) => (
            <tr key={`row-${rowIndex}`}>
              {pad(row).map((cell, cellIndex) => (
                <td key={`${cell}-${cellIndex}`}>
                  <InlineText text={cell} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CourseTableBlock({ rows }) {
  const columns = ["Semester", "Course Code", "Course Title", "Credits", "CIA", "ESE"];

  return (
    <div className="table-scroll">
      <table className="answer-table course-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.semester}-${row.code}-${row.title}-${index}`}>
              <td>{row.semester}</td>
              <td>{row.code || "-"}</td>
              <td>
                <InlineText text={row.title} />
              </td>
              <td>{row.credits}</td>
              <td>{row.cia}</td>
              <td>{row.ese}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AnswerContent({ content }) {
  const blocks = [];
  const lines = normalizeDisplayText(content).split(/\r?\n/);
  let paragraph = [];
  let bullets = [];
  let tableRows = [];
  let courseRows = [];
  let activeSemester = "";

  function flushParagraph() {
    if (!paragraph.length) return;
    blocks.push({ type: "paragraph", text: paragraph.join(" ") });
    paragraph = [];
  }

  function flushBullets() {
    if (!bullets.length) return;
    blocks.push({ type: "bullets", items: bullets });
    bullets = [];
  }

  function flushTable() {
    if (!tableRows.length) return;
    blocks.push({ type: "table", rows: tableRows });
    tableRows = [];
  }

  function flushCourseRows() {
    if (!courseRows.length) return;
    blocks.push({ type: "course-table", rows: courseRows });
    courseRows = [];
  }

  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushBullets();
      flushTable();
      flushCourseRows();
      return;
    }

    if (/^[=-]{4,}$/.test(trimmed)) {
      flushParagraph();
      flushBullets();
      flushTable();
      flushCourseRows();
      return;
    }

    const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushBullets();
      flushTable();
      flushCourseRows();
      blocks.push({
        type: "heading",
        level: Math.min(heading[1].length, 4),
        text: cleanHeadingText(heading[2]),
      });
      return;
    }

    const semesterInfo = extractSemester(trimmed);
    if (semesterInfo) {
      activeSemester = semesterInfo.semester;
      flushParagraph();
      flushBullets();
      flushTable();
      if (!semesterInfo.rest) return;

      const course = parseCourseDetail(semesterInfo.rest, activeSemester);
      if (course) {
        courseRows.push(course);
        return;
      }
      paragraph.push(cleanHeadingText(trimmed));
      return;
    }

    const course = parseCourseDetail(trimmed, activeSemester);
    if (course) {
      flushParagraph();
      flushBullets();
      flushTable();
      courseRows.push(course);
      return;
    }

    if (isPipeTableRow(trimmed)) {
      flushParagraph();
      flushBullets();
      flushCourseRows();
      tableRows.push(trimmed);
      return;
    }

    const bullet = trimmed.match(/^(?:[-*]\s+)(.+)$/);
    if (bullet) {
      flushParagraph();
      flushTable();
      flushCourseRows();
      bullets.push(bullet[1]);
      return;
    }

    flushBullets();
    flushTable();
    flushCourseRows();
    paragraph.push(cleanHeadingText(trimmed));
  });

  flushParagraph();
  flushBullets();
  flushTable();
  flushCourseRows();

  return (
    <div className="answer-content">
      {blocks.map((block, index) =>
        block.type === "bullets" ? (
          <ul key={`bullets-${index}`}>
            {block.items.map((item, itemIndex) => (
              <li key={`${item}-${itemIndex}`}>
                <InlineText text={item} />
              </li>
            ))}
          </ul>
        ) : block.type === "table" ? (
          <TableBlock key={`table-${index}`} rows={block.rows} />
        ) : block.type === "course-table" ? (
          <CourseTableBlock key={`course-table-${index}`} rows={block.rows} />
        ) : block.type === "heading" ? (
          <div className={`answer-heading answer-heading-${block.level}`} key={`heading-${index}`}>
            <InlineText text={block.text} />
          </div>
        ) : (
          <p key={`paragraph-${index}`}>
            <InlineText text={block.text} />
          </p>
        )
      )}
    </div>
  );
}

function SourcePreview({ preview }) {
  const lines = String(preview || "").split(/\r?\n/);
  const hasTableRows = lines.some(isPipeTableRow);
  if (!hasTableRows) {
    return (
      <div className="source-preview">
        <p>{preview}</p>
      </div>
    );
  }

  return (
    <div className="source-preview">
      <AnswerContent content={preview} />
    </div>
  );
}

function SourceList({ sources }) {
  if (!sources?.length) {
    return <p className="muted">No sources returned yet.</p>;
  }

  return (
    <div className="source-list">
      {sources.map((source, index) => (
        <article className="source" key={`${source.url || source.path || index}`}>
          <div className="source-topline">
            <span className="citation">[{source.citation || index + 1}]</span>
            <strong title={source.label || "Untitled source"}>{source.label || "Untitled source"}</strong>
          </div>
          {source.preview && <SourcePreview preview={source.preview} />}
          <div className="source-meta">
            {source.category && <span>{source.category}</span>}
            {typeof source.rerank_score === "number" && (
              <span>score {source.rerank_score.toFixed(2)}</span>
            )}
            {source.url && (
              <a href={source.url} target="_blank" rel="noreferrer">
                Open <ExternalLink size={13} />
              </a>
            )}
          </div>
        </article>
      ))}
    </div>
  );
}

function Message({ message, onCopy }) {
  const isUser = message.role === "user";
  return (
    <article className={`message ${isUser ? "message-user" : "message-assistant"}`}>
      <div className="avatar" aria-hidden="true">
        {isUser ? (
          <MessageSquareText size={18} />
        ) : (
          <img src={CUK_LOGO} alt="" />
        )}
      </div>
      <div className="bubble">
        <div className="message-meta">
          <span>{isUser ? "You" : "CUK Assistant"}</span>
          <time>{formatTime(message.createdAt)}</time>
          {!isUser && (
            <button className="icon-button" type="button" onClick={() => onCopy(message.content)} title="Copy answer">
              <Copy size={15} />
            </button>
          )}
        </div>
        <AnswerContent content={message.content} />
        {!isUser && <SourceList sources={message.sources} />}
      </div>
    </article>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [answerStyle, setAnswerStyle] = useState("balanced");
  const [theme, setTheme] = useState("light");
  const [status, setStatus] = useState({ loading: true, data: null, error: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef(null);

  const history = useMemo(() => buildHistory(messages), [messages]);
  const kbReady = !!status.data?.knowledge_base_ready;
  const generatorReady = status.data?.generator_configured !== false;
  const backendReady = kbReady && generatorReady;
  const generatorProvider = status.data?.generator_provider || "generator";
  const generatorLabel = generatorProvider.charAt(0).toUpperCase() + generatorProvider.slice(1);

  useEffect(() => {
    let cancelled = false;
    let timer;

    async function pollStatus() {
      try {
        const data = await fetchStatus();
        if (!cancelled) setStatus({ loading: false, data, error: "" });
      } catch (err) {
        if (!cancelled) {
          setStatus({ loading: false, data: null, error: err.message });
        }
      } finally {
        if (!cancelled) timer = window.setTimeout(pollStatus, 7000);
      }
    }

    pollStatus();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  const abortRef = useRef(null);

  async function submitQuery(nextQuery = input) {
    const query = nextQuery.trim();
    if (!query || loading) return;

    setError("");
    setInput("");
    const userMessage = {
      role: "user",
      content: query,
      createdAt: new Date(),
    };

    setMessages((current) => [...current, userMessage]);
    setLoading(true);

    try {
      const { answer, sources } = await askQuestion({
        query,
        history,
        answerStyle,
      });
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: answer,
          sources: sources || [],
          createdAt: new Date(),
        },
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    submitQuery();
  }

  async function copyText(text) {
    await navigator.clipboard?.writeText(text);
  }

  return (
    <main className="app-shell" data-theme={theme}>
      <CampusScene3D />
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <img src={CUK_LOGO} alt="Central University of Kashmir logo" />
          </div>
          <div>
            <h1>Central University of Kashmir</h1>
            <p>University knowledge assistant</p>
          </div>
        </div>

        <section className="panel">
          <div className="panel-title">
            <Settings2 size={17} />
            <span>Answer Style</span>
          </div>
          <div className="segmented">
            {ANSWER_STYLES.map((style) => (
              <button
                className={answerStyle === style.id ? "active" : ""}
                key={style.id}
                type="button"
                onClick={() => setAnswerStyle(style.id)}
              >
                {style.label}
              </button>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-title">
            {theme === "dark" ? <Moon size={17} /> : <Sun size={17} />}
            <span>Theme</span>
          </div>
          <div className="theme-toggle" role="group" aria-label="Theme">
            <button
              className={theme === "light" ? "active" : ""}
              type="button"
              onClick={() => setTheme("light")}
            >
              <Sun size={16} />
              Light
            </button>
            <button
              className={theme === "dark" ? "active" : ""}
              type="button"
              onClick={() => setTheme("dark")}
            >
              <Moon size={16} />
              Dark
            </button>
          </div>
        </section>

        <section className="panel status-panel">
          <div className="panel-title">
            {backendReady ? <CheckCircle2 size={17} /> : <CircleAlert size={17} />}
            <span>Backend</span>
          </div>
          <p className={backendReady ? "status-good" : "status-warn"}>
            {status.loading
              ? "Checking API..."
              : backendReady
                ? "Knowledge base ready"
                : kbReady && !generatorReady
                  ? `${generatorLabel} API key missing`
                : status.error || status.data?.message || "Knowledge base not ready"}
          </p>
          {status.data && (
            <dl className="status-grid">
              <div>
                <dt>Chunks</dt>
                <dd title={status.data.vector_db_dir || ""}>
                  {formatChunkCount(status.data.chunk_count)}
                </dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd>{status.data.generator_model || "not set"}</dd>
              </div>
            </dl>
          )}
        </section>

        <button className="clear-button" type="button" onClick={() => setMessages([])}>
          <Eraser size={17} />
          Clear chat
        </button>
      </aside>

      <section className="chat">
        <header className="chat-header">
          <div className="header-brand">
            <img src={CUK_LOGO} alt="Central University of Kashmir logo" />
            <div>
              <span className="eyebrow">Central University of Kashmir</span>
            </div>
          </div>
          <div className="header-actions">
            <button
              className="header-theme-button"
              type="button"
              onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
              title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
              aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            >
              {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
            </button>
            <div className="header-chip">
              <BookOpen size={16} />
              {messages.length} messages
            </div>
          </div>
        </header>

        <div className="conversation">
          {messages.length === 0 && (
            <div className="empty-state">
              <h3>Start with a question</h3>
              <div className="prompt-grid">
                {QUICK_PROMPTS.map((prompt) => (
                  <button key={prompt} type="button" onClick={() => submitQuery(prompt)}>
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message, index) => (
            <Message key={`${message.role}-${index}`} message={message} onCopy={copyText} />
          ))}

          {loading && (
            <div className="loading-row">
              <Loader2 size={18} />
              Searching documents and drafting an answer...
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {error && <div className="error-bar">{error}</div>}

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            aria-label="Ask a question"
            placeholder="Ask about admissions, notices, faculty, departments, contacts..."
            rows={1}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submitQuery();
              }
            }}
          />
          <button className="send-button" type="submit" disabled={loading || !input.trim()} title="Send question">
            {loading ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
            <span>Send</span>
          </button>
        </form>
      </section>
    </main>
  );
}
