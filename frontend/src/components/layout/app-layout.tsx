import { Link, NavLink, Outlet } from "react-router-dom";

import { cn } from "@/lib/utils";

export function AppLayout() {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-border/70 bg-background/90 backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
          <Link to="/chat" className="text-sm font-semibold tracking-wide text-primary">
            ELTE RAG Assistant
          </Link>
          <nav className="flex items-center gap-2">
            <NavLink
              to="/chat"
              className={({ isActive }) =>
                cn(
                  "rounded-md px-3 py-1.5 text-sm",
                  isActive ? "bg-primary text-primary-foreground" : "hover:bg-muted",
                )
              }
            >
              Chat
            </NavLink>
            <NavLink
              to="/admin"
              className={({ isActive }) =>
                cn(
                  "rounded-md px-3 py-1.5 text-sm",
                  isActive ? "bg-primary text-primary-foreground" : "hover:bg-muted",
                )
              }
            >
              Admin
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-6xl p-4 sm:p-6">
        <Outlet />
      </main>
    </div>
  );
}
