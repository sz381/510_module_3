import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Seattle nightly price estimator",
  description: "Estimate a nightly price using Seattle Airbnb listings from January 2016.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return <html lang="en"><body>{children}</body></html>;
}
