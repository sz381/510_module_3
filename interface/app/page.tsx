"use client";

import { useEffect, useState, type FormEvent } from "react";

type Metadata = { room_types: string[]; neighborhood_groups: string[] };

function SelectField({ label, name, options, initial }: {
  label: string; name: string; options: (string | number)[]; initial: string | number;
}) {
  return (
    <label>{label}
      <select name={name} defaultValue={initial} required>
        {options.map(option => <option key={option} value={option}>{option}</option>)}
      </select>
    </label>
  );
}

const guests = Array.from({ length: 16 }, (_, i) => i + 1);
const bedrooms = Array.from({ length: 8 }, (_, i) => i);
const bathrooms = Array.from({ length: 17 }, (_, i) => i / 2);
const dollars = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

export default function Home() {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const [price, setPrice] = useState<number | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const response = await fetch("/api/metadata", {
          cache: "no-store",
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(10000)]),
        });
        if (!response.ok) throw new Error();
        const data = await response.json();
        if (![data.room_types, data.neighborhood_groups].every(options =>
          Array.isArray(options) && options.length && options.every(option => typeof option === "string")
        )) throw new Error();
        setMetadata(data);
      } catch {
        if (!controller.signal.aborted) setError("Listing options could not be loaded. Check that the API is running and try again.");
      }
    }
    void load();
    return () => controller.abort();
  }, [attempt]);

  async function estimate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const values = new FormData(event.currentTarget);
    setPending(true);
    setError("");
    setPrice(null);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(10000),
        body: JSON.stringify({
          room_type: values.get("room_type"),
          neighborhood_group: values.get("neighborhood_group"),
          accommodates: Number(values.get("accommodates")),
          bedrooms: Number(values.get("bedrooms")),
          bathrooms: Number(values.get("bathrooms")),
        }),
      });
      if (!response.ok) throw new Error();
      const data = await response.json();
      if (!Number.isFinite(data.nightly_price_usd)) throw new Error();
      setPrice(data.nightly_price_usd);
    } catch {
      setError("The price estimate is unavailable. Check that the API is running and try again.");
    } finally {
      setPending(false);
    }
  }

  return (
    <main className="estimator">
      <header>
        <h1>Seattle nightly price estimator</h1>
        <p>Based on Seattle Airbnb listings from January 2016.</p>
      </header>
      <form onSubmit={estimate} onChange={() => { setPrice(null); setError(""); }}>
        {metadata ? (
          <fieldset disabled={pending}>
            <legend className="sr-only">Listing details</legend>
            <div className="choices">
              <SelectField label="Room type" name="room_type" options={metadata.room_types} initial="Entire home/apt" />
              <SelectField label="Neighborhood" name="neighborhood_group" options={metadata.neighborhood_groups} initial="Capitol Hill" />
            </div>
            <div className="capacity">
              <SelectField label="Guests" name="accommodates" options={guests} initial={4} />
              <SelectField label="Bedrooms" name="bedrooms" options={bedrooms} initial={2} />
              <SelectField label="Bathrooms" name="bathrooms" options={bathrooms} initial={1.5} />
            </div>
            <button type="submit">{pending ? "Estimating…" : "Estimate nightly price"}</button>
          </fieldset>
        ) : error ? (
          <button type="button" onClick={() => { setError(""); setAttempt(n => n + 1); }}>Retry loading options</button>
        ) : <p role="status">Loading listing options…</p>}
        {error && <p role="alert">{error}</p>}
      </form>
      <section className="estimate-result" aria-live="polite" aria-atomic="true">
        <h2>Estimated nightly price</h2>
        {price !== null ? <p><strong>{dollars.format(price)}</strong> USD</p> : <p>Submit your listing details to see an estimate.</p>}
      </section>
    </main>
  );
}
